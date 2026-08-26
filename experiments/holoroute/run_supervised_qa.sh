#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
BASE_OUT=${REPO}/experiments/holoroute/outputs/routing_fingerprint_qa
OUT=${REPO}/experiments/holoroute/outputs/routing_fingerprint_supervised_qa
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}

cd "${REPO}" || exit 1
mkdir -p "${BASE_OUT}" "${OUT}"

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
if [ -n "${TRAIN_LIMIT}" ]; then
  TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
fi
if [ -n "${TEST_LIMIT}" ]; then
  TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")
fi

if [ ! -f "${BASE_OUT}/method.pt" ] || [ ! -f "${BASE_OUT}/reference.npz" ]; then
  echo
  echo "[0/3] Fit the unlabeled feature normalization"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.holoroute.run fit \
    --train "${TRAIN_SPLIT}" \
    --checkpoint "${BASE_OUT}/method.pt" \
    --reference "${BASE_OUT}/reference.npz" \
    --task QA \
    --device "${DEVICE}" \
    "${TRAIN_LIMIT_ARGUMENT[@]}" || exit $?
fi

echo
if [ -f "${OUT}/probe.npz" ]; then
  echo "[1/3] Reuse the existing labeled linear probe"
else
  echo "[1/3] Fit a labeled linear probe on the same node features"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.holoroute.run probe-fit \
    --train "${TRAIN_SPLIT}" \
    --checkpoint "${BASE_OUT}/method.pt" \
    --reference "${BASE_OUT}/reference.npz" \
    --probe "${OUT}/probe.npz" \
    --task QA \
    --device "${DEVICE}" \
    "${TRAIN_LIMIT_ARGUMENT[@]}" || exit $?
fi

echo
if [ -f "${OUT}/scores.npz" ]; then
  echo "[2/3] Reuse the existing test-node scores"
else
  echo "[2/3] Score test nodes with the labeled linear probe"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.holoroute.run probe-score \
    --test "${TEST_SPLIT}" \
    --checkpoint "${BASE_OUT}/method.pt" \
    --reference "${BASE_OUT}/reference.npz" \
    --probe "${OUT}/probe.npz" \
    --output "${OUT}/scores.npz" \
    --task QA \
    --device "${DEVICE}" \
    "${TEST_LIMIT_ARGUMENT[@]}" || exit $?
fi

echo
echo "[3/3] Evaluate the labeled probe"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m experiments.holoroute.run evaluate \
  --test "${TEST_SPLIT}" \
  --scores "${OUT}/scores.npz" \
  --output "${OUT}/evaluation" \
  --device "${DEVICE}" || exit $?

echo
echo "Finished."
echo "Probe: ${OUT}/probe.npz"
echo "Metrics: ${OUT}/evaluation/evaluation.json"
