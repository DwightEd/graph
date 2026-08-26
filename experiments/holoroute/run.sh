#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-"${ROOT}/experiments/holoroute/outputs/routing_fingerprint"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}

if [ -z "${TRAIN_SPLIT}" ] || [ -z "${TEST_SPLIT}" ]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
if [ -n "${TRAIN_LIMIT}" ]; then
  TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
fi
if [ -n "${TEST_LIMIT}" ]; then
  TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")
fi

echo
echo "[1/3] Build train node features and fit normal subspace"
"${PYTHON}" -m experiments.holoroute.run fit \
  --train "${TRAIN_SPLIT}" \
  --checkpoint "${OUT}/method.pt" \
  --reference "${OUT}/reference.npz" \
  --task "${TASK}" \
  --device "${DEVICE}" \
  "${TRAIN_LIMIT_ARGUMENT[@]}" || exit $?

echo
echo "[2/3] Build test node features, score and export graphs"
"${PYTHON}" -m experiments.holoroute.run score \
  --test "${TEST_SPLIT}" \
  --checkpoint "${OUT}/method.pt" \
  --reference "${OUT}/reference.npz" \
  --output "${OUT}/scores.npz" \
  --task "${TASK}" \
  --device "${DEVICE}" \
  "${TEST_LIMIT_ARGUMENT[@]}" || exit $?

echo
echo "[3/3] Evaluate"
"${PYTHON}" -m experiments.holoroute.run evaluate \
  --test "${TEST_SPLIT}" \
  --scores "${OUT}/scores.npz" \
  --output "${OUT}/evaluation" \
  --device "${DEVICE}" || exit $?

echo
echo "Finished."
echo "Output: ${OUT}"
echo "Graphs: ${OUT}/graphs"
echo "Metrics: ${OUT}/evaluation/evaluation.json"
