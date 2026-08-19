#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

ROOT="${ROOT:-$PROJECT_ROOT/../data/RAGTruth/llama31_8b}"
OUT="${OUT:-$PROJECT_ROOT/experiments/token_routing_basin/outputs/run_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "$PROJECT_ROOT/.audit_envs/research/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.audit_envs/research/bin/python"
  else
    PYTHON="$(command -v python)"
  fi
fi
if [[ -n "$LIMIT" && ! "$LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LIMIT must be a positive integer" >&2
  exit 2
fi
for split in train test; do
  if [[ ! -f "$ROOT/$split/manifest.json" ]]; then
    echo "Missing canonical manifest: $ROOT/$split/manifest.json" >&2
    exit 2
  fi
done

mkdir -p "$OUT/logs"
limit_args=()
if [[ -n "$LIMIT" ]]; then
  limit_args=(--limit "$LIMIT")
fi

if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  "$PYTHON" -m pytest -q \
    tests/test_token_routing_basin.py \
    tests/test_token_routing_basin_cli.py
fi

"$PYTHON" -m experiments.token_routing_basin.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  "${limit_args[@]}" 2>&1 | tee "$OUT/logs/fit.log"
test -f "$OUT/reference.npz"

"$PYTHON" -m experiments.token_routing_basin.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${limit_args[@]}" 2>&1 | tee "$OUT/logs/score.log"
test -f "$OUT/test_scores.npz"

"$PYTHON" -m experiments.token_routing_basin.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output-dir "$OUT/evaluation" \
  --device cpu 2>&1 | tee "$OUT/logs/evaluate.log"
test -f "$OUT/evaluation/report.json"

printf 'Completed token routing basin run: %s\n' "$OUT"
