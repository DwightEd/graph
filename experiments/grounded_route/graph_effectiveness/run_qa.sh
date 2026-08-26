#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
GROUND_OUT=${REPO}/experiments/grounded_route/outputs/qa
OUT=${GROUND_OUT}/graph_effectiveness

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
CALIBRATION_INDEX="${GROUND_OUT}/calibration/index.npz" \
GRAPH_INDEX="${GROUND_OUT}/test/index.npz" \
UNSUPERVISED_SCORES="${GROUND_OUT}/scores.npz" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${OUT}" \
DEVICE=cuda \
FOLDS=5 \
EPOCHS=20 \
SEEDS="20260825 20260826 20260827" \
bash experiments/grounded_route/graph_effectiveness/run.sh
