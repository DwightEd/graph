#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
POSITIVE_EDGES_PER_GRAPH=${POSITIVE_EDGES_PER_GRAPH:-4096}
HOLDOUT_FRACTION=${HOLDOUT_FRACTION:-0.15}
NEGATIVE_COUNT=${NEGATIVE_COUNT:-1}
NEGATIVE_ATTEMPT_FACTOR=${NEGATIVE_ATTEMPT_FACTOR:-8}
LAYOUT_ROWS_PER_GRAPH=${LAYOUT_ROWS_PER_GRAPH:-32}
LAYOUT_ROWS_PER_BATCH=${LAYOUT_ROWS_PER_BATCH:-64}
LAYOUT_MIN_MASS=${LAYOUT_MIN_MASS:-0.0001}
LAYOUT_MAX_ELEMENTS=${LAYOUT_MAX_ELEMENTS:-8000000}
LAYOUT_MAX_WORK_ELEMENTS=${LAYOUT_MAX_WORK_ELEMENTS:-250000000}
LAYOUT_ORDER=${LAYOUT_ORDER:-ordered}
INCIDENCE_DROPOUT=${INCIDENCE_DROPOUT:-0.0}
HEAD_DROPOUT=${HEAD_DROPOUT:-0.0}
FLOW_WEIGHT=${FLOW_WEIGHT:-0.0}
LAYOUT_WEIGHT=${LAYOUT_WEIGHT:-0.0}
VARIANCE_WEIGHT=${VARIANCE_WEIGHT:-0.05}
RESIDUAL_WEIGHT=${RESIDUAL_WEIGHT:-1.0}
SLOT_DIM=${SLOT_DIM:-16}
EDGE_HIDDEN_DIM=${EDGE_HIDDEN_DIM:-64}
LATENT_MODE=${LATENT_MODE:-deterministic}
VAE_EXPORT=${VAE_EXPORT:-mean_logvar}
KL_WEIGHT=${KL_WEIGHT:-0.001}
KL_FREE_BITS=${KL_FREE_BITS:-0.01}
KL_WARMUP_EPOCHS=${KL_WARMUP_EPOCHS:-4}
VARIANT=${VARIANT:-real}
SEED=${SEED:-20260827}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
DEFAULT_RUN_NAME="endpoint_recovery_${VARIANT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_ep${EPOCHS}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_tl${TRAIN_LIMIT:-all}_xl${TEST_LIMIT:-all}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_pe${POSITIVE_EDGES_PER_GRAPH}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_hf${HOLDOUT_FRACTION}_neg${NEGATIVE_COUNT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_na${NEGATIVE_ATTEMPT_FACTOR}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_id${INCIDENCE_DROPOUT}_hd${HEAD_DROPOUT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_fw${FLOW_WEIGHT}_lw${LAYOUT_WEIGHT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_lo${LAYOUT_ORDER}_rw${RESIDUAL_WEIGHT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_lrows${LAYOUT_ROWS_PER_GRAPH}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_vw${VARIANCE_WEIGHT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_latent${LATENT_MODE}_export${VAE_EXPORT}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_sd${SLOT_DIM}_eh${EDGE_HIDDEN_DIM}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_kl${KL_WEIGHT}_fb${KL_FREE_BITS}"
DEFAULT_RUN_NAME="${DEFAULT_RUN_NAME}_kw${KL_WARMUP_EPOCHS}_seed${SEED}"
RUN_NAME=${RUN_NAME:-${DEFAULT_RUN_NAME}}
OUT=${OUT:-${ROOT}/experiments/directed_route_hypergraph/outputs/${TASK,,}/${RUN_NAME}}
EVALUATE=${EVALUATE:-1}
START_STAGE=${START_STAGE:-1}

if [ -z "${TRAIN_SPLIT:-}" ] || [ -z "${TEST_SPLIT:-}" ]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must be set."
  exit 1
fi
if ! [[ "${START_STAGE}" =~ ^[1-5]$ ]]; then
  echo "START_STAGE must be an integer from 1 to 5."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
[ -n "${TRAIN_LIMIT}" ] && TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
[ -n "${TEST_LIMIT}" ] && TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")

source_fingerprint() {
  {
    find experiments/directed_route_hypergraph experiments/grounded_route \
      -type f \( -name '*.py' -o -name '*.sh' \) -print0
    printf '%s\0' research_dataset.py experiment_protocol.py
  } | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}

FINGERPRINT_FILE="${OUT}/source_fingerprint.txt"
CURRENT_FINGERPRINT="$(source_fingerprint)"
if [ "${START_STAGE}" = "1" ]; then
  printf '%s\n' "${CURRENT_FINGERPRINT}" > "${FINGERPRINT_FILE}"
else
  if [ ! -f "${FINGERPRINT_FILE}" ]; then
    echo "Cannot resume: ${FINGERPRINT_FILE} is missing."
    exit 1
  fi
  SAVED_FINGERPRINT="$(tr -d '[:space:]' < "${FINGERPRINT_FILE}")"
  if [ "${CURRENT_FINGERPRINT}" != "${SAVED_FINGERPRINT}" ]; then
    echo "Cannot resume: source files differ from the code that started this run."
    echo "Use the matching checkout for this checkpoint or start a new output directory."
    exit 1
  fi
  if [ ! -f "${OUT}/model.pt" ]; then
    echo "Cannot resume: ${OUT}/model.pt is missing."
    exit 1
  fi
fi

verify_source() {
  local current
  current="$(source_fingerprint)"
  if [ "${current}" != "$(tr -d '[:space:]' < "${FINGERPRINT_FILE}")" ]; then
    echo
    echo "Source files changed while the pipeline was running."
    echo "The checkpoint was preserved, but later stages were stopped to avoid loading it with a different model definition."
    exit 2
  fi
}

run_stage() {
  local stage="$1"
  local label="$2"
  shift 2
  if [ "${START_STAGE}" -le "${stage}" ]; then
    verify_source
    echo
    echo "${label}"
    "$@" || exit $?
    verify_source
  else
    echo
    echo "${label} -- skipped (START_STAGE=${START_STAGE})"
  fi
}

run_stage 1 "[1/5] Fit label-free directed hypergraph encoder" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run fit \
  --train "${TRAIN_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --task "${TASK}" --epochs "${EPOCHS}" \
  --positive-edges-per-graph "${POSITIVE_EDGES_PER_GRAPH}" \
  --holdout-fraction "${HOLDOUT_FRACTION}" \
  --negative-count "${NEGATIVE_COUNT}" \
  --negative-attempt-factor "${NEGATIVE_ATTEMPT_FACTOR}" \
  --variant "${VARIANT}" \
  --layout-rows-per-graph "${LAYOUT_ROWS_PER_GRAPH}" \
  --layout-rows-per-batch "${LAYOUT_ROWS_PER_BATCH}" \
  --layout-min-mass "${LAYOUT_MIN_MASS}" \
  --layout-max-elements "${LAYOUT_MAX_ELEMENTS}" \
  --layout-max-work-elements "${LAYOUT_MAX_WORK_ELEMENTS}" \
  --layout-order "${LAYOUT_ORDER}" \
  --incidence-dropout "${INCIDENCE_DROPOUT}" \
  --head-dropout "${HEAD_DROPOUT}" --flow-weight "${FLOW_WEIGHT}" \
  --layout-weight "${LAYOUT_WEIGHT}" \
  --variance-weight "${VARIANCE_WEIGHT}" \
  --residual-weight "${RESIDUAL_WEIGHT}" \
  --slot-dim "${SLOT_DIM}" --edge-hidden-dim "${EDGE_HIDDEN_DIM}" \
  --latent-mode "${LATENT_MODE}" --vae-export "${VAE_EXPORT}" \
  --kl-weight "${KL_WEIGHT}" --kl-free-bits "${KL_FREE_BITS}" \
  --kl-warmup-epochs "${KL_WARMUP_EPOCHS}" \
  --seed "${SEED}" --device "${DEVICE}" "${TRAIN_LIMIT_ARGUMENT[@]}"

run_stage 2 "[2/5] Export calibration node embeddings" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run encode \
  --data "${TRAIN_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --output "${OUT}/calibration" --scope calibration \
  --task "${TASK}" --device "${DEVICE}"

run_stage 3 "[3/5] Export test node embeddings" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run encode \
  --data "${TEST_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --output "${OUT}/test" --scope all --task "${TASK}" \
  --device "${DEVICE}" "${TEST_LIMIT_ARGUMENT[@]}"

run_stage 4 "[4/5] Fit node-only PCA-kNN and freeze scores" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run detect \
  --calibration "${OUT}/calibration/index.npz" \
  --test "${OUT}/test/index.npz" \
  --reference "${OUT}/detector.npz" --scores "${OUT}/scores.npz" \
  --seed "${SEED}"

if [ "${EVALUATE}" = "1" ]; then
  run_stage 5 "[5/5] Evaluate frozen token scores" \
    "${PYTHON}" -m experiments.directed_route_hypergraph.run evaluate \
    --test "${TEST_SPLIT}" --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" --seed "${SEED}"
else
  echo
  echo "[5/5] Labels remain closed"
fi

echo
echo "Finished: ${OUT}"
