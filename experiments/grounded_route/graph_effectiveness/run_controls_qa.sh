#!/usr/bin/env bash

set -euo pipefail

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
BASE=${BASE:-${REPO}/experiments/grounded_route/outputs/qa_controls}
ENCODER_SEED=${ENCODER_SEED:-20260825}

cd "${REPO}" || exit 1

for VIEW in real no_message endpoint_rewire weight_shuffle; do
  VARIANT="${VIEW}"
  MESSAGE_MODE=neighbor
  if [[ "${VIEW}" == "no_message" ]]; then
    VARIANT=real
    MESSAGE_MODE=row_local
  fi
  # Keep the pipeline's generation guard permissive; the audit below enforces
  # the registered 10% global and 80%-of-samples intervention thresholds.
  CUDA_VISIBLE_DEVICES=0 \
  PYTHON=python \
  TRAIN_SPLIT="${TRAIN_SPLIT}" \
  TEST_SPLIT="${TEST_SPLIT}" \
  OUT="${BASE}/${VIEW}" \
  TASK=QA \
  VARIANT="${VARIANT}" \
  MESSAGE_MODE="${MESSAGE_MODE}" \
  DEVICE=cuda \
  EPOCHS=8 \
  SEED="${ENCODER_SEED}" \
  MINIMUM_CHANGED_FRACTION=0.01 \
  EVALUATE=0 \
  bash experiments/grounded_route/run.sh
done

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
CALIBRATION_INDEX="${BASE}/real/calibration/index.npz" \
GRAPH_INDEX="${BASE}/real/test/index.npz" \
UNSUPERVISED_SCORES="${BASE}/real/scores.npz" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${BASE}/graph_effectiveness" \
DEVICE=cuda \
FOLDS=5 \
EPOCHS=20 \
SEEDS="20260825 20260826 20260827" \
CONTROL_ARGUMENTS="--control no_message ${BASE}/no_message/calibration/index.npz ${BASE}/no_message/test/index.npz --control endpoint_rewire ${BASE}/endpoint_rewire/calibration/index.npz ${BASE}/endpoint_rewire/test/index.npz --control weight_shuffle ${BASE}/weight_shuffle/calibration/index.npz ${BASE}/weight_shuffle/test/index.npz" \
bash experiments/grounded_route/graph_effectiveness/run.sh
