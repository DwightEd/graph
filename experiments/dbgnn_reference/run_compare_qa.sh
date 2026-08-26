#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
GRAPH_OUT=${REPO}/experiments/grounded_route/outputs/qa
TRAIN_INDEX=${GRAPH_OUT}/calibration/index.npz
TEST_INDEX=${GRAPH_OUT}/test/index.npz
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
BASE_OUT=${REPO}/experiments/dbgnn_reference/outputs/qa_compare

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
TRAIN_INDEX="${TRAIN_INDEX}" \
TEST_INDEX="${TEST_INDEX}" \
TEST_SPLIT="${TEST_SPLIT}" \
BASE_OUT="${BASE_OUT}" \
DEVICE=cuda \
EPOCHS=8 \
DIAGNOSTIC_EPOCHS=20 \
bash experiments/dbgnn_reference/run_compare.sh
