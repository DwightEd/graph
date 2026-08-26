#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
MODEL=${MODEL:-dbgnn}
HIGHER_ORDER_MODE=${HIGHER_ORDER_MODE:-causal}
EPOCHS=${EPOCHS:-8}
SEED=${SEED:-20260826}
EVALUATE=${EVALUATE:-1}
RESUME=${RESUME:-0}
START_STAGE=${START_STAGE:-1}
OUT=${OUT:-"${ROOT}/experiments/dbgnn_reference/outputs/${MODEL}"}

if [ -z "${TRAIN_INDEX:-}" ] || [ -z "${TEST_INDEX:-}" ]; then
  echo "TRAIN_INDEX and TEST_INDEX must point to saved token-graph bundles."
  exit 1
fi
if [ "${EVALUATE}" = "1" ] && [ -z "${TEST_SPLIT:-}" ]; then
  echo "TEST_SPLIT is required when EVALUATE=1."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

if [ "${RESUME}" = "1" ]; then
  [ -f "${OUT}/checkpoint.pt" ] && START_STAGE=2
  [ -f "${OUT}/calibration/index.npz" ] && START_STAGE=3
  [ -f "${OUT}/test/index.npz" ] && START_STAGE=4
  [ -f "${OUT}/scores.npz" ] && START_STAGE=5
fi

echo "Starting ${MODEL} from stage ${START_STAGE}"

if [ "${START_STAGE}" -le 1 ]; then
  echo "[1/5] Fit ${MODEL} with label-free endpoint prediction"
  "${PYTHON}" -m experiments.dbgnn_reference.run fit \
    --train-index "${TRAIN_INDEX}" \
    --checkpoint "${OUT}/checkpoint.pt" \
    --encoder "${MODEL}" \
    --higher-order-mode "${HIGHER_ORDER_MODE}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" || exit $?
fi

if [ "${START_STAGE}" -le 2 ]; then
  echo "[2/5] Export calibration node embeddings"
  "${PYTHON}" -m experiments.dbgnn_reference.run encode \
    --index "${TRAIN_INDEX}" \
    --checkpoint "${OUT}/checkpoint.pt" \
    --output "${OUT}/calibration" \
    --scope calibration \
    --device "${DEVICE}" || exit $?
fi

if [ "${START_STAGE}" -le 3 ]; then
  echo "[3/5] Export test node embeddings"
  "${PYTHON}" -m experiments.dbgnn_reference.run encode \
    --index "${TEST_INDEX}" \
    --checkpoint "${OUT}/checkpoint.pt" \
    --output "${OUT}/test" \
    --scope all \
    --device "${DEVICE}" || exit $?
fi

if [ "${START_STAGE}" -le 4 ]; then
  echo "[4/5] Fit PCA-kNN on node embeddings and freeze scores"
  "${PYTHON}" -m experiments.dbgnn_reference.run detect \
    --calibration "${OUT}/calibration/index.npz" \
    --test "${OUT}/test/index.npz" \
    --reference "${OUT}/detector.npz" \
    --scores "${OUT}/scores.npz" \
    --seed "${SEED}" || exit $?
fi

if [ "${EVALUATE}" = "1" ]; then
  echo "[5/5] Open labels only for post-hoc evaluation"
  "${PYTHON}" -m experiments.dbgnn_reference.run evaluate \
    --test "${TEST_SPLIT}" \
    --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" \
    --seed "${SEED}" || exit $?
else
  echo "[5/5] Keep labels closed"
fi

echo "Finished: ${OUT}"
