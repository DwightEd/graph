#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-D:/projects/python_projects/research/data/RAGTruth/llama31_8b}
OUT=${OUT:-experiments/attention_phenomenology/outputs/head_model_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-cpu}
PYTHON=${PYTHON:-python}

LIMITS=()
[[ -n "${TRAIN_LIMIT:-}" ]] && LIMITS+=(--train-limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && LIMITS+=(--test-limit "$TEST_LIMIT")
ABLATIONS=()
[[ "${MASK_RESPONSE_REUSE:-0}" == "1" ]] && ABLATIONS+=(--mask-response-reuse)

"$PYTHON" -u -m experiments.attention_phenomenology.main train-head-model \
  --train-split "$ROOT/train" \
  --test-split "$ROOT/test" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --validation-fraction "${VALIDATION_FRACTION:-0.2}" \
  --reuse-top-k "${REUSE_TOP_K:-5}" \
  --recent-response-tokens "${RECENT_RESPONSE_TOKENS:-4}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --hidden-dim "${HIDDEN_DIM:-16}" \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --dropout "${DROPOUT:-0}" \
  --forecast-weight "${FORECAST_WEIGHT:-0.5}" \
  --patience "${PATIENCE:-5}" \
  --seed "${SEED:-20260820}" \
  "${LIMITS[@]}" \
  "${ABLATIONS[@]}"

printf '\nDone: %s\n' "$OUT/evaluation.json"
