#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-"${ROOT}/experiments/holoroute/outputs/run"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
MODEL=${MODEL:-holoroute}

if [ -z "${TRAIN_SPLIT}" ] || [ -z "${TEST_SPLIT}" ]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

TRAIN_COMMAND=train
SCORE_COMMAND=score
if [ "${MODEL}" = "flat1024" ]; then
  TRAIN_COMMAND=flat-train
  SCORE_COMMAND=flat-score
fi

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
if [ -n "${TRAIN_LIMIT}" ]; then
  TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
fi
if [ -n "${TEST_LIMIT}" ]; then
  TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")
fi

echo
echo "[1/3] Train ${MODEL}"
"${PYTHON}" -m experiments.holoroute.run "${TRAIN_COMMAND}" \
  --train "${TRAIN_SPLIT}" \
  --checkpoint "${OUT}/model.pt" \
  --reference "${OUT}/reference.npz" \
  --task "${TASK}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  "${TRAIN_LIMIT_ARGUMENT[@]}" || exit $?

echo
echo "[2/3] Score ${MODEL}"
"${PYTHON}" -m experiments.holoroute.run "${SCORE_COMMAND}" \
  --test "${TEST_SPLIT}" \
  --checkpoint "${OUT}/model.pt" \
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
echo "Metrics: ${OUT}/evaluation/evaluation.json"
