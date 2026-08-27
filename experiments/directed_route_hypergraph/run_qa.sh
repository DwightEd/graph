#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
TRAIN_SPLIT=${TRAIN_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
VARIANT=${VARIANT:-real}
SEED=${SEED:-20260827}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT=${OUT:-${REPO}/experiments/directed_route_hypergraph/outputs/${TASK,,}/${VARIANT}_seed${SEED}}

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
PYTHON="${PYTHON}" \
TRAIN_SPLIT="${TRAIN_SPLIT}" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${OUT}" \
TASK="${TASK}" \
DEVICE="${DEVICE}" \
EPOCHS=${EPOCHS:-8} \
ROWS_PER_GRAPH=${ROWS_PER_GRAPH:-256} \
VARIANT="${VARIANT}" \
SEED="${SEED}" \
TRAIN_LIMIT=${TRAIN_LIMIT:-} \
TEST_LIMIT=${TEST_LIMIT:-} \
EVALUATE=${EVALUATE:-1} \
bash experiments/directed_route_hypergraph/run.sh
