#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

TRAIN_EXTRA=()
TEST_EXTRA=()
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  if (( TRAIN_LIMIT < 8 )); then
    echo "TRAIN_LIMIT must be at least 8; use 64 or more for a meaningful smoke test" >&2
    exit 2
  fi
  TRAIN_EXTRA=(--limit "$TRAIN_LIMIT")
fi
if [[ -n "${TEST_LIMIT:-}" ]]; then
  if (( TEST_LIMIT < 1 )); then
    echo "TEST_LIMIT must be positive" >&2
    exit 2
  fi
  TEST_EXTRA=(--limit "$TEST_LIMIT")
fi

if [[ -n "${TRAIN_LIMIT:-}" || -n "${TEST_LIMIT:-}" ]]; then
  RUN_NAME="smoke_train${TRAIN_LIMIT:-all}_test${TEST_LIMIT:-all}"
  EPOCHS_DEFAULT=1
else
  RUN_NAME=full
  EPOCHS_DEFAULT=2
fi
OUT=${OUT:-experiments/causal_multiplex_flow/outputs/$RUN_NAME}

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi
mkdir -p "$OUT"

echo "dataset_root=$ROOT"
echo "output=$OUT"
echo "device=$DEVICE"
echo "train_limit=${TRAIN_LIMIT:-all}"
echo "test_limit=${TEST_LIMIT:-all}"
echo "epochs=${EPOCHS:-$EPOCHS_DEFAULT}"
echo "max_prompt_events=${MAX_PROMPT_EVENTS:-16}"
echo "max_rr_events=${MAX_RR_EVENTS:-32}"
echo "negatives=${NEGATIVES:-8}"
echo "hidden_dim=${HIDDEN_DIM:-64}"

printf '\n[1/3] fit label-free causal multiplex router\n'
"$PYTHON" -u -m experiments.causal_multiplex_flow.main fit \
  --train-split "$ROOT/train" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --max-prompt-events "${MAX_PROMPT_EVENTS:-16}" \
  --max-rr-events "${MAX_RR_EVENTS:-32}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --channel-embedding-dim "${CHANNEL_EMBEDDING_DIM:-8}" \
  --relation-embedding-dim "${RELATION_EMBEDDING_DIM:-4}" \
  --lag-frequencies "${LAG_FREQUENCIES:-4}" \
  --negatives "${NEGATIVES:-8}" \
  --dropout "${DROPOUT:-0.10}" \
  --weight-loss-weight "${WEIGHT_LOSS_WEIGHT:-0.10}" \
  --epochs "${EPOCHS:-$EPOCHS_DEFAULT}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --weight-decay "${WEIGHT_DECAY:-0.00001}" \
  --gradient-clip "${GRADIENT_CLIP:-1.0}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --seed "${SEED:-20260817}" \
  "${TRAIN_EXTRA[@]}"

printf '\n[2/3] freeze test routing-surprise scores without labels\n'
"$PYTHON" -u -m experiments.causal_multiplex_flow.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${TEST_EXTRA[@]}"

printf '\n[3/3] post-hoc token-label evaluation\n'
"$PYTHON" -u -m experiments.causal_multiplex_flow.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device cpu

echo "done: $OUT/evaluation.json"
