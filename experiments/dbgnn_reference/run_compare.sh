#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-8}
DIAGNOSTIC_EPOCHS=${DIAGNOSTIC_EPOCHS:-20}
SEED=${SEED:-20260826}
BASE_OUT=${BASE_OUT:-"${ROOT}/experiments/dbgnn_reference/outputs/compare"}

if [[ -z "${TRAIN_INDEX:-}" || -z "${TEST_INDEX:-}" || -z "${TEST_SPLIT:-}" ]]; then
  echo "TRAIN_INDEX, TEST_INDEX and TEST_SPLIT must be set."
  exit 1
fi

for RUN in "gcn:gcn:no_transition" "dbgnn_no_transition:dbgnn:no_transition" "dbgnn:dbgnn:causal"; do
  IFS=: read -r NAME MODEL HIGHER_ORDER_MODE <<< "${RUN}"
  PYTHON="${PYTHON}" \
  TRAIN_INDEX="${TRAIN_INDEX}" \
  TEST_INDEX="${TEST_INDEX}" \
  OUT="${BASE_OUT}/${NAME}" \
  MODEL="${MODEL}" \
  HIGHER_ORDER_MODE="${HIGHER_ORDER_MODE}" \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  SEED="${SEED}" \
  EVALUATE=0 \
  bash "${ROOT}/experiments/dbgnn_reference/run.sh"
done

"${PYTHON}" -m experiments.dbgnn_reference.diagnostics \
  --gcn-calibration "${BASE_OUT}/gcn/calibration/index.npz" \
  --gcn-test "${BASE_OUT}/gcn/test/index.npz" \
  --no-transition-calibration "${BASE_OUT}/dbgnn_no_transition/calibration/index.npz" \
  --no-transition-test "${BASE_OUT}/dbgnn_no_transition/test/index.npz" \
  --dbgnn-calibration "${BASE_OUT}/dbgnn/calibration/index.npz" \
  --dbgnn-test "${BASE_OUT}/dbgnn/test/index.npz" \
  --test "${TEST_SPLIT}" \
  --output "${BASE_OUT}/diagnostics" \
  --device "${DEVICE}" \
  --epochs "${DIAGNOSTIC_EPOCHS}" \
  --seeds "${SEED}" \
  --seed "${SEED}"

echo "Finished comparison: ${BASE_OUT}/diagnostics/report.json"
