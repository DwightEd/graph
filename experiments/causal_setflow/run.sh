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
[[ -n "${TRAIN_LIMIT:-}" ]] && TRAIN_EXTRA=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && TEST_EXTRA=(--limit "$TEST_LIMIT")
[[ "${ACTIVATION_CHECKPOINTING:-1}" == "0" ]] && CHECKPOINT_EXTRA=(--disable-activation-checkpointing)
[[ "${CUDA_MEMORY_PROFILE:-1}" == "0" ]] && MEMORY_PROFILE_EXTRA=(--disable-cuda-memory-profile)

if [[ -n "${TRAIN_LIMIT:-}" || -n "${TEST_LIMIT:-}" ]]; then
  RUN_NAME="smoke_train${TRAIN_LIMIT:-all}_test${TEST_LIMIT:-all}"
  EPOCHS_DEFAULT=1
  MIN_CONDITION_ROWS_DEFAULT=8
  GRADIENT_ACCUMULATION_DEFAULT=1
else
  RUN_NAME=full
  EPOCHS_DEFAULT=5
  MIN_CONDITION_ROWS_DEFAULT=32
  GRADIENT_ACCUMULATION_DEFAULT=2
fi

OUT=${OUT:-experiments/causal_setflow/outputs/mechanism_guided/$RUN_NAME}
if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi
if [[ -e "$OUT/model.pt" || -e "$OUT/reference.npz" || -e "$OUT/test_scores.npz" || -e "$OUT/evaluation.json" ]]; then
  echo "output already contains frozen MG-CASF artifacts; choose a fresh OUT: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"

cat <<EOF
dataset_root=$ROOT
output=$OUT
device=$DEVICE
precision=${PRECISION:-auto}
epochs=${EPOCHS:-$EPOCHS_DEFAULT}
hidden_dim=${HIDDEN_DIM:-64}
max_route_sources=${MAX_ROUTE_SOURCES:-32}
max_memory_sources=${MAX_MEMORY_SOURCES:-16}
corruption_bank=collapse,localize,freeze,homogenize,self_reinforce
materialize_query_chunk_size=${MATERIALIZE_QUERY_CHUNK_SIZE:-64}
set_row_chunk_size=${SET_ROW_CHUNK_SIZE:-4096}
mixer_token_chunk_size=${MIXER_TOKEN_CHUNK_SIZE:-512}
activation_checkpointing=${ACTIVATION_CHECKPOINTING:-1}
EOF

printf '\n[1/3] label-free mechanism-guided Set-Flow training and calibration\n'
"$PYTHON" -u -m experiments.causal_setflow.main fit \
  --train-split "$ROOT/train" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --max-route-sources "${MAX_ROUTE_SOURCES:-32}" \
  --max-memory-sources "${MAX_MEMORY_SOURCES:-16}" \
  --route-mass-coverage "${ROUTE_MASS_COVERAGE:-0.98}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --materialize-query-chunk-size "${MATERIALIZE_QUERY_CHUNK_SIZE:-64}" \
  --hidden-dim "${HIDDEN_DIM:-64}" \
  --scalar-fourier-dim "${SCALAR_FOURIER_DIM:-16}" \
  --set-heads "${SET_HEADS:-4}" \
  --induced-points "${INDUCED_POINTS:-8}" \
  --set-blocks "${SET_BLOCKS:-2}" \
  --head-mixer-layers "${HEAD_MIXER_LAYERS:-2}" \
  --depth-mixer-layers "${DEPTH_MIXER_LAYERS:-2}" \
  --energy-hidden-multiplier "${ENERGY_HIDDEN_MULTIPLIER:-2}" \
  --projector-hidden-multiplier "${PROJECTOR_HIDDEN_MULTIPLIER:-2}" \
  --set-row-chunk-size "${SET_ROW_CHUNK_SIZE:-4096}" \
  --mixer-token-chunk-size "${MIXER_TOKEN_CHUNK_SIZE:-512}" \
  --dropout "${DROPOUT:-0.10}" \
  --token-span-min "${TOKEN_SPAN_MIN:-4}" \
  --token-span-max "${TOKEN_SPAN_MAX:-24}" \
  --layer-span-min "${LAYER_SPAN_MIN:-4}" \
  --layer-span-max "${LAYER_SPAN_MAX:-12}" \
  --selected-head-fraction "${SELECTED_HEAD_FRACTION:-0.50}" \
  --collapse-power "${COLLAPSE_POWER:-4.0}" \
  --self-reinforce-power "${SELF_REINFORCE_POWER:-2.0}" \
  --locality-window "${LOCALITY_WINDOW:-4}" \
  --corruption-margin "${CORRUPTION_MARGIN:-1.0}" \
  --clean-keep-fraction "${CLEAN_KEEP_FRACTION:-0.90}" \
  --epochs "${EPOCHS:-$EPOCHS_DEFAULT}" \
  --learning-rate "${LEARNING_RATE:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-$GRADIENT_ACCUMULATION_DEFAULT}" \
  --gradient-clip-norm "${GRADIENT_CLIP_NORM:-1.0}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --ema-momentum "${EMA_MOMENTUM:-0.996}" \
  --clean-energy-weight "${CLEAN_ENERGY_WEIGHT:-1.0}" \
  --corrupt-energy-weight "${CORRUPT_ENERGY_WEIGHT:-1.0}" \
  --ranking-weight "${RANKING_WEIGHT:-1.0}" \
  --type-weight "${TYPE_WEIGHT:-0.50}" \
  --clean-recovery-weight "${CLEAN_RECOVERY_WEIGHT:-1.0}" \
  --context-recovery-weight "${CONTEXT_RECOVERY_WEIGHT:-0.50}" \
  --variance-weight "${VARIANCE_WEIGHT:-1.0}" \
  --covariance-weight "${COVARIANCE_WEIGHT:-0.04}" \
  --precision "${PRECISION:-auto}" \
  --seed "${SEED:-20260818}" \
  --min-condition-rows "${MIN_CONDITION_ROWS:-$MIN_CONDITION_ROWS_DEFAULT}" \
  "${CHECKPOINT_EXTRA[@]}" \
  "${MEMORY_PROFILE_EXTRA[@]}" \
  "${TRAIN_EXTRA[@]}"

printf '\n[2/3] freeze held-out MG-CASF energies without labels\n'
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