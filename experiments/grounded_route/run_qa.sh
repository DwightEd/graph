#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
OUT=${REPO}/experiments/grounded_route/outputs/qa

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
TRAIN_SPLIT="${TRAIN_SPLIT}" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${OUT}" \
TASK=QA \
VARIANT=real \
DEVICE=cuda \
EPOCHS=8 \
bash experiments/grounded_route/run.sh
