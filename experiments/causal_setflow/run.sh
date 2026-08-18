#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TRAIN_EXTRA=()
TEST_EXTRA=()
CHECKPOINT_EXTRA=()
MEMORY_PROFILE_EXTRA=()
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  TRAIN_EXTRA=(--limit "$TRAIN_LIMIT")
fi
if [[ -n "${TEST_LIMIT:-}" ]]; then
  TEST_EXTRA=(--limit "$TEST_LIMIT")
fi
if [[ "${ACTIVATION_CHECKPOINTING:-1}" == "0" ]]; then
  CHECKPOINT_EXTRA=(--disable-activation-checkpointing)
fi
if [[ "${CUDA_MEMORY_PROFILE:-1}" == "0" ]]; then
  MEMORY_PROFILE_EXTRA=(--disable-cuda-memory-profile)
fi

if [[ -n "${TRAIN_LIMIT:-}" || -n "${TEST_LIMIT:-}" ]]; then
  RUN_NAME="smoke_train${TRAIN_LIMIT:-all}_test${TEST_LIMIT:-all}"
  EPOCHS_DEFAULT=1
  REFERENCE_PER_SAMPLE_DEFAULT=4
  MIN_CONDITION_ROWS_DEFAULT=8
  GRADIENT_ACCUMULATION_DEFAULT=1
else
  RUN_NAME=full
  EPOCHS_DEFAULT=3
  REFERENCE_PER_SAMPLE_DEFAULT=8
  MIN_CONDITION_ROWS_DEFAULT=32
  GRADIENT_ACCUMULATION_DEFAULT=4
fi

# Smoke and full runs use the same source-set and model architecture. Only the
# number of samples/epochs and calibration support differ.
HIDDEN_DIM_DEFAULT=64
DETERMINISTIC_MASKS_DEFAULT=4
OUT=${OUT:-experiments/causal_setflow/outputs/v1/$RUN_NAME}
MAX_ROUTE_SOURCES_VALUE=${MAX_ROUTE_SOURCES:-${MAX_SOURCES:-32}}
MAX_MEMORY_SOURCES_VALUE=${MAX_MEMORY_SOURCES:-${MAX_SOURCES:-16}}

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi
if [[ -e "$OUT/model.pt" || -e "$OUT/reference.npz" || -e "$OUT/test_scores.npz" || -e "$OUT/evaluation.json" ]]; then
  echo "output already contains frozen CASF artifacts; choose a fresh OUT: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT"

echo "dataset_root=$ROOT"
echo "output=$OUT"
echo "device=$DEVICE"
echo "precision=${PRECISION:-auto}"
echo "epochs=${EPOCHS:-$EPOCHS_DEFAULT}"
echo "hidden_dim=${HIDDEN_DIM:-$HIDDEN_DIM_DEFAULT}"
echo "max_route_sources=$MAX_ROUTE_SOURCES_VALUE"
echo "max_memory_sources=$MAX_MEMORY_SOURCES_VALUE"
echo "materialize_query_chunk_size=${MATERIALIZE_QUERY_CHUNK_SIZE:-64}"
echo "set_row_chunk_size=${SET_ROW_CHUNK_SIZE:-4096}"
echo "mixer_token_chunk_size=${MIXER_TOKEN_CHUNK_SIZE:-512}"
echo "activation_checkpointing=${ACTIVATION_CHECKPOINTING:-1}"

printf '\n[1/3] label-free masked Causal Attention Set-Flow training and calibration\n'
"$PYTHON" -u -m experiments.causal_setflow.main fit \
  --train-split "$ROOT/train" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --max-route-sources "$MAX_ROUTE_SOURCES_VALUE" \
  --max-memory-sources "$MAX_MEMORY_SOURCES_VALUE" \
  --route-mass-coverage "${ROUTE_MASS_COVERAGE:-0.98}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --materialize-query-chunk-size "${MATERIALIZE_QUERY_CHUNK_SIZE:-64}" \
  --hidden-dim "${HIDDEN_DIM:-$HIDDEN_DIM_DEFAULT}" \
  --scalar-fourier-dim "${SCALAR_FOURIER_DIM:-16}" \
  --set-heads "${SET_HEADS:-4}" \
  --induced-points "${INDUCED_POINTS:-8}" \
  --set-blocks "${SET_BLOCKS:-2}" \
  --head-mixer-layers "${HEAD_MIXER_LAYERS:-2}" \
  --depth-mixer-layers "${DEPTH_MIXER_LAYERS:-2}" \
  --set-row-chunk-size "${SET_ROW_CHUNK_SIZE:-4096}" \
  --mixer-token-chunk-size "${MIXER_TOKEN_CHUNK_SIZE:-512}" \
  --dropout "${DROPOUT:-0.10}" \
  --element-mask-probability "${ELEMENT_MASK_PROBABILITY:-0.20}" \
  --head-mask-probability "${HEAD_MASK_PROBABILITY:-0.20}" \
  --layer-mask-probability "${LAYER_MASK_PROBABILITY:-0.15}" \
  --epochs "${EPOCHS:-$EPOCHS_DEFAULT}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-$GRADIENT_ACCUMULATION_DEFAULT}" \
  --gradient-clip-norm "${GRADIENT_CLIP_NORM:-1.0}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-$REFERENCE_PER_SAMPLE_DEFAULT}" \
  --latent-trim-fraction "${LATENT_TRIM_FRACTION:-0.90}" \
  --deterministic-masks "${DETERMINISTIC_MASKS:-$DETERMINISTIC_MASKS_DEFAULT}" \
  --precision "${PRECISION:-auto}" \
  --seed "${SEED:-20260818}" \
  --min-condition-rows "${MIN_CONDITION_ROWS:-$MIN_CONDITION_ROWS_DEFAULT}" \
  "${CHECKPOINT_EXTRA[@]}" \
  "${MEMORY_PROFILE_EXTRA[@]}" \
  "${TRAIN_EXTRA[@]}"

printf '\n[2/3] freeze held-out CASF token scores without labels\n'
"$PYTHON" -u -m experiments.causal_setflow.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${TEST_EXTRA[@]}"

printf '\n[3/3] post-hoc token-label evaluation\n'
"$PYTHON" -u -m experiments.causal_setflow.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device cpu

echo "done: $OUT/evaluation.json"
echo "reference: $OUT/reference.npz"
echo "scores:    $OUT/test_scores.npz"