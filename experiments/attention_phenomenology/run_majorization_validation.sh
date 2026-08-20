#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
OUT=${OUT:-experiments/attention_phenomenology/outputs/majorization_validation}
DEVICE=${DEVICE:-cpu}
PYTHON=${PYTHON:-python}

LIMITS=()
[[ -n "${FIT_LIMIT:-}" ]] && LIMITS+=(--fit-limit "$FIT_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && LIMITS+=(--test-limit "$TEST_LIMIT")

"$PYTHON" -u -m experiments.attention_phenomenology.main validate-majorization \
  --train-split "$ROOT/train" \
  --test-split "$ROOT/test" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --history-decay "${HISTORY_DECAY:-0.9}" \
  --majorization-tolerance "${MAJORIZATION_TOLERANCE:-0.000001}" \
  --fit-tokens-per-sample "${FIT_TOKENS_PER_SAMPLE:-128}" \
  --minimum-scale "${MINIMUM_SCALE:-0.01}" \
  --maximum-standardized-value "${MAXIMUM_STANDARDIZED_VALUE:-10}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-200}" \
  --seed "${SEED:-20260820}" \
  "${LIMITS[@]}"

printf '\nDone: %s\n' "$OUT/evaluation.json"
