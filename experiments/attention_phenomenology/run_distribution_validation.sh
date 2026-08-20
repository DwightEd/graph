#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
OUT=${OUT:-experiments/attention_phenomenology/outputs/dirichlet_validation}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

LIMITS=()
PSEUDOCOUNT_ARGS=()
IFS=',' read -ra PSEUDOCOUNTS <<< "${PSEUDOCOUNTS:-0.000001,0.0001,0.001}"
for VALUE in "${PSEUDOCOUNTS[@]}"; do
  PSEUDOCOUNT_ARGS+=(--pseudocount "$VALUE")
done
[[ -n "${FIT_LIMIT:-}" ]] && LIMITS+=(--fit-limit "$FIT_LIMIT")
[[ -n "${VALIDATION_LIMIT:-}" ]] && LIMITS+=(--validation-limit "$VALIDATION_LIMIT")

"$PYTHON" -u -m experiments.attention_phenomenology.main validate-distributions \
  --fit-split "$ROOT/train" \
  --validation-split "$ROOT/test" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --fit-reservoir-rows "${FIT_RESERVOIR_ROWS:-1024}" \
  --validation-reservoir-rows "${VALIDATION_RESERVOIR_ROWS:-1024}" \
  --minimum-group-rows "${MINIMUM_GROUP_ROWS:-128}" \
  "${PSEUDOCOUNT_ARGS[@]}" \
  --simulation-rows "${SIMULATION_ROWS:-4096}" \
  --causal-position-bins "${CAUSAL_POSITION_BINS:-10}" \
  --recent-response-tokens "${RECENT_RESPONSE_TOKENS:-4}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --seed "${SEED:-20260820}" \
  "${LIMITS[@]}"

printf '\nDone: %s\n' "$OUT/summary.json"
