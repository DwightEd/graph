#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORMAL_ROOT="${FORMAL_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
CANONICAL_ROOT="${CANONICAL_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b}"
GRAPH_ROOT="${GRAPH_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels}"
DEVICE="${DEVICE:-cuda}"
K_PROMPT="${K_PROMPT:-8}"
K_HISTORY="${K_HISTORY:-8}"

CANONICAL_ROOT="$(realpath -m -- "$CANONICAL_ROOT")"
GRAPH_ROOT="$(realpath -m -- "$GRAPH_ROOT")"
[[ "$CANONICAL_ROOT" != "$GRAPH_ROOT" && "$CANONICAL_ROOT" != "$GRAPH_ROOT/"* && "$GRAPH_ROOT" != "$CANONICAL_ROOT/"* ]] || {
  printf 'Canonical and graph outputs must not overlap.\n' >&2
  exit 1
}
[[ ! -e "$CANONICAL_ROOT" && ! -L "$CANONICAL_ROOT" ]] || {
  printf 'Canonical output already exists: %s\n' "$CANONICAL_ROOT" >&2
  exit 1
}
[[ ! -e "$GRAPH_ROOT" && ! -L "$GRAPH_ROOT" ]] || {
  printf 'Graph output already exists: %s\n' "$GRAPH_ROOT" >&2
  exit 1
}

CANONICAL_STAGING="${CANONICAL_ROOT}.staging.$$"
GRAPH_STAGING="${GRAPH_ROOT}.staging.$$"
[[ "$CANONICAL_STAGING" != "$GRAPH_STAGING" ]] || {
  printf 'Canonical and graph outputs must differ.\n' >&2
  exit 1
}
[[ ! -e "$CANONICAL_STAGING" && ! -L "$CANONICAL_STAGING" && ! -e "$GRAPH_STAGING" && ! -L "$GRAPH_STAGING" ]] || {
  printf 'Staging output already exists for PID %s.\n' "$$" >&2
  exit 1
}

cleanup_staging() {
  local status="$?"
  if [[ "$status" -ne 0 ]]; then
    rm -rf -- "$CANONICAL_STAGING" "$GRAPH_STAGING"
  fi
}
trap cleanup_staging EXIT

"$PYTHON_BIN" "$REPO_DIR/main.py" archive-attention \
  --formal-root "$FORMAL_ROOT" \
  --output-root "$CANONICAL_STAGING"
"$PYTHON_BIN" "$REPO_DIR/main.py" verify-attention --archive-root "$CANONICAL_STAGING"

for split in train test; do
  "$PYTHON_BIN" "$REPO_DIR/main.py" build \
    --cache-dir "$CANONICAL_STAGING/$split" \
    --output-dir "$GRAPH_STAGING/$split" \
    --kind relation_topk_channels \
    --k-prompt "$K_PROMPT" \
    --k-history "$K_HISTORY" \
    --device "$DEVICE"
done

[[ ! -e "$CANONICAL_ROOT" && ! -L "$CANONICAL_ROOT" && ! -e "$GRAPH_ROOT" && ! -L "$GRAPH_ROOT" ]] || {
  printf 'Final output appeared before publication.\n' >&2
  exit 1
}
mv -T -- "$CANONICAL_STAGING" "$CANONICAL_ROOT"
if ! mv -T -- "$GRAPH_STAGING" "$GRAPH_ROOT"; then
  mv -T -- "$CANONICAL_ROOT" "$CANONICAL_STAGING"
  exit 1
fi
