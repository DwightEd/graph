#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT=${VARIANT:-real}
MESSAGE_MODE=${MESSAGE_MODE:-neighbor}
if [[ -z "${OUT:-}" ]]; then
  OUT="${ROOT}/experiments/grounded_route/outputs/${VARIANT}"
  if [[ "${MESSAGE_MODE}" != "neighbor" ]]; then
    OUT="${OUT}_${MESSAGE_MODE}"
  fi
fi
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
SEED=${SEED:-20260825}
MINIMUM_CHANGED_FRACTION=${MINIMUM_CHANGED_FRACTION:-0.01}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
EVALUATE=${EVALUATE:-1}

if [[ -z "${TRAIN_SPLIT:-}" || -z "${TEST_SPLIT:-}" ]]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}"

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
if [[ -n "${TRAIN_LIMIT}" ]]; then
  TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
fi
if [[ -n "${TEST_LIMIT}" ]]; then
  TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")
fi

echo "[1/7] Build train graph spec"
"${PYTHON}" -m experiments.grounded_route.run build \
  --data "${TRAIN_SPLIT}" \
  --output "${OUT}/train_graph.json" \
  --task "${TASK}" \
  "${TRAIN_LIMIT_ARGUMENT[@]}"

echo "[2/7] Fit grounded-route encoder"
"${PYTHON}" -m experiments.grounded_route.run fit \
  --spec "${OUT}/train_graph.json" \
  --checkpoint "${OUT}/model.pt" \
  --epochs "${EPOCHS}" \
  --variant "${VARIANT}" \
  --message-mode "${MESSAGE_MODE}" \
  --seed "${SEED}" \
  --minimum-changed-fraction "${MINIMUM_CHANGED_FRACTION}" \
  --device "${DEVICE}"

echo "[3/7] Encode detector calibration nodes"
"${PYTHON}" -m experiments.grounded_route.run encode \
  --spec "${OUT}/train_graph.json" \
  --checkpoint "${OUT}/model.pt" \
  --scope calibration \
  --output "${OUT}/calibration" \
  --variant "${VARIANT}" \
  --message-mode "${MESSAGE_MODE}" \
  --device "${DEVICE}"

echo "[4/7] Build test graph spec"
"${PYTHON}" -m experiments.grounded_route.run build \
  --data "${TEST_SPLIT}" \
  --output "${OUT}/test_graph.json" \
  --task "${TASK}" \
  "${TEST_LIMIT_ARGUMENT[@]}"

echo "[5/7] Encode test token graphs"
"${PYTHON}" -m experiments.grounded_route.run encode \
  --spec "${OUT}/test_graph.json" \
  --checkpoint "${OUT}/model.pt" \
  --scope all \
  --output "${OUT}/test" \
  --variant "${VARIANT}" \
  --message-mode "${MESSAGE_MODE}" \
  --device "${DEVICE}"

echo "[6/7] Fit PCA-kNN and score test nodes"
"${PYTHON}" -m experiments.grounded_route.run detect \
  --calibration "${OUT}/calibration/index.npz" \
  --test "${OUT}/test/index.npz" \
  --reference "${OUT}/detector.npz" \
  --scores "${OUT}/scores.npz"

if [[ "${EVALUATE}" == "1" ]]; then
  echo "[7/7] Evaluate frozen scores"
  "${PYTHON}" -m experiments.grounded_route.run evaluate \
    --test "${TEST_SPLIT}" \
    --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" \
    --device cpu
else
  echo "[7/7] Keep labels closed for downstream frozen-score audit"
fi

echo "Finished: ${OUT}"
