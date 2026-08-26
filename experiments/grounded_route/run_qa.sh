#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
OUT=${OUT:-${REPO}/experiments/grounded_route/outputs/qa}
EPOCHS=${EPOCHS:-8}
SEED=${SEED:-20260825}
MESSAGE_MODE=${MESSAGE_MODE:-neighbor}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
TRAIN_SPLIT="${TRAIN_SPLIT}" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${OUT}" \
TASK=QA \
VARIANT=real \
MESSAGE_MODE="${MESSAGE_MODE}" \
DEVICE=cuda \
EPOCHS="${EPOCHS}" \
SEED="${SEED}" \
TRAIN_LIMIT="${TRAIN_LIMIT}" \
TEST_LIMIT="${TEST_LIMIT}" \
bash experiments/grounded_route/run.sh
