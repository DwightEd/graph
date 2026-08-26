#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
BASE=${OUT:-${REPO}/experiments/grounded_route/outputs/qa_attention_row}
TEST_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test

cd "${REPO}" || exit 1

CALIBRATION_INDEX="${BASE}/calibration/index.npz" \
TEST_INDEX="${BASE}/test/index.npz" \
TEST_ROOT="${TEST_ROOT}" \
OUT="${BASE}/evaluation" \
DEVICE=cuda \
SEEDS=20260826 \
bash experiments/grounded_route/evaluation/run.sh
