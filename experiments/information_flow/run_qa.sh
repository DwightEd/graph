#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
GCN=${GCN:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare/gcn}
TEST_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
OUT=${OUT:-${REPO}/experiments/information_flow/outputs/qa}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-20260827}
LIMIT=${LIMIT:-}

cd "${REPO}" || exit 1
mkdir -p "${OUT}"

LIMIT_ARGUMENT=()
[ -n "${LIMIT}" ] && LIMIT_ARGUMENT=(--limit "${LIMIT}")

python -m experiments.information_flow.run \
  --source-index "${GCN}/calibration/index.npz" \
  --output "${OUT}/sketch/calibration" \
  --head-mode sketch --seed "${SEED}" --device "${DEVICE}" \
  "${LIMIT_ARGUMENT[@]}" || exit $?

python -m experiments.information_flow.run \
  --source-index "${GCN}/test/index.npz" \
  --output "${OUT}/sketch/test" \
  --head-mode sketch --seed "${SEED}" --device "${DEVICE}" \
  "${LIMIT_ARGUMENT[@]}" || exit $?

python -m experiments.information_flow.run \
  --source-index "${GCN}/calibration/index.npz" \
  --output "${OUT}/mean/calibration" \
  --head-mode mean --seed "${SEED}" --device "${DEVICE}" \
  "${LIMIT_ARGUMENT[@]}" || exit $?

python -m experiments.information_flow.run \
  --source-index "${GCN}/test/index.npz" \
  --output "${OUT}/mean/test" \
  --head-mode mean --seed "${SEED}" --device "${DEVICE}" \
  "${LIMIT_ARGUMENT[@]}" || exit $?

CONTROL_ARGUMENTS="--control gcn ${GCN}/calibration/index.npz ${GCN}/test/index.npz --control mean_head ${OUT}/mean/calibration/index.npz ${OUT}/mean/test/index.npz" \
CALIBRATION_INDEX="${OUT}/sketch/calibration/index.npz" \
TEST_INDEX="${OUT}/sketch/test/index.npz" \
TEST_ROOT="${TEST_ROOT}" \
OUT="${OUT}/evaluation" \
DEVICE="${DEVICE}" \
bash experiments/grounded_route/evaluation/run.sh || exit $?

printf '\nFinished: %s\n' "${OUT}/evaluation/report.json"
