#!/usr/bin/env bash
# All tasks, all generators, both official splits. No four-answer or label-balanced cap.
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-benchmark \
  --dataset "${RAGTRUTH_DATASET:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset}" \
  --model "${MODEL:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}" \
  --output outputs/native_support_ragtruth_all/source_first_v1 \
  --resume "$@"
