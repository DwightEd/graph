#!/usr/bin/env bash

set -euo pipefail

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
SOURCE=${SOURCE:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare/gcn}
TRAIN_SPLIT=${TRAIN_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
OUT=${OUT:-${REPO}/experiments/dbgnn_reference/outputs/gcn_node_data_qa.npz}
PYTHON=${PYTHON:-python}

cd "${REPO}"

"${PYTHON}" -m experiments.dbgnn_reference.export_node_data \
  --calibration-index "${SOURCE}/calibration/index.npz" \
  --test-index "${SOURCE}/test/index.npz" \
  --calibration-split "${TRAIN_SPLIT}" \
  --test-split "${TEST_SPLIT}" \
  --output "${OUT}"

du -h "${OUT}"
