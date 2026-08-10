#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAFE_ROOT="/share/home/tm902089733300000/a903202310/lys"
CANONICAL_ROOT="${CANONICAL_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b}"
GRAPH_ROOT="${GRAPH_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels}"
LEGACY_GRAPH_ROOT="/share/home/tm902089733300000/a903202310/lys/data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05"
FORMAL_ROOT="/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
LEGACY_GRAPH_RELATIVE="data/feature_extraction/ragtruth_original_attribute_graphs/fresh_attention_c8847872bedf_20260731T074520Z_p876_tau0p05"
FORMAL_RELATIVE="research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
DRY_RUN="${DRY_RUN:-1}"
DELETE_FORMAL="${DELETE_FORMAL:-0}"
CONFIRM_DELETE="${CONFIRM_DELETE:-}"

[[ -d "$CANONICAL_ROOT" ]] || {
  printf 'Canonical archive is not a directory: %s\n' "$CANONICAL_ROOT" >&2
  exit 1
}
"$PYTHON_BIN" - "$CANONICAL_ROOT" "$GRAPH_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

canonical_root, graph_root = map(Path, sys.argv[1:])
for split in ("train", "test"):
    canonical = canonical_root / split
    canonical_manifest = json.loads((canonical / "manifest.json").read_text(encoding="utf-8"))
    count = canonical_manifest.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise SystemExit(f"Canonical {split} count must be positive.")

for split in ("train", "test"):
    canonical = canonical_root / split
    graph = graph_root / split
    canonical_manifest = json.loads((canonical / "manifest.json").read_text(encoding="utf-8"))
    graph_manifest = json.loads((graph / "manifest.json").read_text(encoding="utf-8"))
    count = canonical_manifest["count"]
    rows = [json.loads(line) for line in (graph / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_ids = [row["sample_id"] for row in rows]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (graph_manifest.get("count") != count
            or len(rows) != count or len(set(sample_ids)) != count
            or graph_manifest.get("kind") != "relation_topk_channels"
            or graph_manifest.get("input_manifest_sha256") != digest(canonical / "manifest.json")
            or graph_manifest.get("input_index_sha256") != digest(canonical / "index.jsonl")
            or graph_manifest.get("index_sha256") != digest(graph / "index.jsonl")):
        raise SystemExit(f"Replacement graph provenance is invalid for {split}.")
PY
for split in train test; do
  "$PYTHON_BIN" "$REPO_DIR/main.py" verify-attention --archive-root "$CANONICAL_ROOT/$split"
done

canonical_resolved="$(realpath -e -- "$CANONICAL_ROOT")"
graph_resolved="$(realpath -e -- "$GRAPH_ROOT")"
safe_root_resolved="$(realpath -e -- "$SAFE_ROOT")"
resolved_targets=()

verify_formal_provenance() {
  "$PYTHON_BIN" - "$CANONICAL_ROOT" "$FORMAL_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

canonical_root, formal_root = map(Path, sys.argv[1:])
for split in ("train", "test"):
    canonical_manifest = canonical_root / split / "manifest.json"
    formal_manifest = formal_root / split / "manifest.json"
    expected = json.loads(canonical_manifest.read_text(encoding="utf-8")).get("source_manifest_sha256")
    actual = hashlib.sha256(formal_manifest.read_bytes()).hexdigest()
    if expected != actual:
        print(f"Formal manifest SHA256 does not match canonical source for {split}.", file=sys.stderr)
        raise SystemExit(1)
PY
}

collect_target() {
  local name="$1"
  local target="$2"
  local relative="$3"
  local expected_lexical
  local target_lexical
  local expected_resolved
  local resolved_target
  local component
  local path

  expected_lexical="$(realpath -ms -- "$SAFE_ROOT/$relative")"
  target_lexical="$(realpath -ms -- "$target")"
  [[ "$target_lexical" == "$expected_lexical" ]] || {
    printf 'Target is not the fixed %s path: %s\n' "$name" "$target" >&2
    exit 1
  }

  path="$safe_root_resolved"
  IFS=/ read -r -a components <<< "$relative"
  for component in "${components[@]}"; do
    path="$path/$component"
    [[ ! -L "$path" ]] || {
      printf 'Target or ancestor is a symbolic link: %s\n' "$path" >&2
      exit 1
    }
  done
  if [[ ! -e "$target" ]]; then
    printf 'Already absent: %s\n' "$expected_lexical"
    return
  fi
  [[ -d "$target" ]] || {
    printf 'Target is not a directory: %s\n' "$target" >&2
    exit 1
  }

  expected_resolved="$safe_root_resolved/$relative"
  resolved_target="$(realpath -e -- "$target")"
  [[ "$resolved_target" == "$expected_resolved" ]] || {
    printf 'Target resolves outside its fixed path: %s\n' "$resolved_target" >&2
    exit 1
  }
  [[ "$resolved_target" != "$canonical_resolved" ]] || {
    printf 'Target is the canonical archive: %s\n' "$resolved_target" >&2
    exit 1
  }
  [[ "$resolved_target" != "$canonical_resolved/"* && "$canonical_resolved" != "$resolved_target/"* ]] || {
    printf 'Target overlaps the canonical archive: %s\n' "$resolved_target" >&2
    exit 1
  }
  [[ "$resolved_target" != "$graph_resolved" ]] || {
    printf 'Target is the replacement graph: %s\n' "$resolved_target" >&2
    exit 1
  }
  [[ "$resolved_target" != "$graph_resolved/"* && "$graph_resolved" != "$resolved_target/"* ]] || {
    printf 'Target overlaps the replacement graph: %s\n' "$resolved_target" >&2
    exit 1
  }
  resolved_targets+=("$resolved_target")
}

if [[ -e "$FORMAL_ROOT" ]]; then
  verify_formal_provenance
fi
collect_target "legacy graph" "$LEGACY_GRAPH_ROOT" "$LEGACY_GRAPH_RELATIVE"
if [[ "$DELETE_FORMAL" == "1" ]]; then
  collect_target "formal cache" "$FORMAL_ROOT" "$FORMAL_RELATIVE"
fi

if [[ "$DRY_RUN" != "0" ]]; then
  for target in "${resolved_targets[@]}"; do
    printf 'Would delete: %s\n' "$target"
    du -sh -- "$target"
  done
  exit 0
fi

[[ "$CONFIRM_DELETE" == "DELETE_RAGTRUTH_LEGACY" ]] || {
  printf 'Set CONFIRM_DELETE=DELETE_RAGTRUTH_LEGACY to delete targets.\n' >&2
  exit 1
}

for target in "${resolved_targets[@]}"; do
  printf 'Deleting: %s\n' "$target"
  rm -rf -- "$target"
done
