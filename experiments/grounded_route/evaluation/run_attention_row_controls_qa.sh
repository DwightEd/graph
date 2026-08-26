#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
BASE=${BASE:-${REPO}/experiments/grounded_route/outputs/qa_attention_row_controls}
SEED=${SEED:-20260825}
EPOCHS=${EPOCHS:-8}

cd "${REPO}" || exit 1

for VIEW in real no_message endpoint_rewire weight_shuffle; do
  VARIANT=${VIEW}
  MESSAGE_MODE=neighbor
  if [ "${VIEW}" = "no_message" ]; then
    VARIANT=real
    MESSAGE_MODE=row_local
  fi

  TRAIN_SPLIT="${TRAIN_ROOT}" \
  TEST_SPLIT="${TEST_ROOT}" \
  OUT="${BASE}/${VIEW}" \
  TASK=QA \
  VARIANT="${VARIANT}" \
  MESSAGE_MODE="${MESSAGE_MODE}" \
  DEVICE=cuda \
  EPOCHS="${EPOCHS}" \
  SEED="${SEED}" \
  EVALUATE=0 \
  bash experiments/grounded_route/run.sh || exit $?
done

CONTROL_ARGUMENTS="--control no_message ${BASE}/no_message/calibration/index.npz ${BASE}/no_message/test/index.npz --control endpoint_rewire ${BASE}/endpoint_rewire/calibration/index.npz ${BASE}/endpoint_rewire/test/index.npz --control weight_shuffle ${BASE}/weight_shuffle/calibration/index.npz ${BASE}/weight_shuffle/test/index.npz" \
CALIBRATION_INDEX="${BASE}/real/calibration/index.npz" \
TEST_INDEX="${BASE}/real/test/index.npz" \
TEST_ROOT="${TEST_ROOT}" \
OUT="${BASE}/evaluation" \
DEVICE=cuda \
SEEDS=20260826 \
bash experiments/grounded_route/evaluation/run.sh
