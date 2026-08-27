#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
ROWS_PER_GRAPH=${ROWS_PER_GRAPH:-256}
LAYOUT_ROWS_PER_GRAPH=${LAYOUT_ROWS_PER_GRAPH:-32}
LAYOUT_ROWS_PER_BATCH=${LAYOUT_ROWS_PER_BATCH:-64}
LAYOUT_MIN_MASS=${LAYOUT_MIN_MASS:-0.0001}
LAYOUT_MAX_ELEMENTS=${LAYOUT_MAX_ELEMENTS:-8000000}
LAYOUT_MAX_WORK_ELEMENTS=${LAYOUT_MAX_WORK_ELEMENTS:-250000000}
LAYOUT_ORDER=${LAYOUT_ORDER:-ordered}
INCIDENCE_DROPOUT=${INCIDENCE_DROPOUT:-0.15}
HEAD_DROPOUT=${HEAD_DROPOUT:-0.05}
FLOW_WEIGHT=${FLOW_WEIGHT:-0.5}
LAYOUT_WEIGHT=${LAYOUT_WEIGHT:-0.25}
RESIDUAL_WEIGHT=${RESIDUAL_WEIGHT:-1.0}
VARIANT=${VARIANT:-real}
SEED=${SEED:-20260827}
RUN_NAME=${RUN_NAME:-${LAYOUT_ORDER}_layout_${VARIANT}_lr${LAYOUT_ROWS_PER_GRAPH}_fw${FLOW_WEIGHT}_lw${LAYOUT_WEIGHT}_rw${RESIDUAL_WEIGHT}_seed${SEED}}
OUT=${OUT:-${ROOT}/experiments/directed_route_hypergraph/outputs/${TASK,,}/${RUN_NAME}}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
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
  --rows-per-graph "${ROWS_PER_GRAPH}" --variant "${VARIANT}" \
  --layout-rows-per-graph "${LAYOUT_ROWS_PER_GRAPH}" \
  --layout-rows-per-batch "${LAYOUT_ROWS_PER_BATCH}" \
  --layout-min-mass "${LAYOUT_MIN_MASS}" \
  --layout-max-elements "${LAYOUT_MAX_ELEMENTS}" \
  --layout-max-work-elements "${LAYOUT_MAX_WORK_ELEMENTS}" \
  --layout-order "${LAYOUT_ORDER}" \
  --incidence-dropout "${INCIDENCE_DROPOUT}" \
  --head-dropout "${HEAD_DROPOUT}" --flow-weight "${FLOW_WEIGHT}" \
  --layout-weight "${LAYOUT_WEIGHT}" \
  --residual-weight "${RESIDUAL_WEIGHT}" \
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
