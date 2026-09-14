#!/usr/bin/env bash
# Explicitly SUPERVISED historical S10 baseline; not the default detector.
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
exec "$BASE/conda_envs/research/bin/python" -m structural_detector.experiment \
 --features ../reanchor/outputs/s10_structural_features_20260914_v1 \
 --annotations "$BASE/data/RAGTruth/dataset/response.jsonl" \
 --output outputs/s10_structural_fusion_20260914_v1
