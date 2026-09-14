#!/usr/bin/env bash
set -euo pipefail
cd /share/home/tm902089733300000/a903202310/lys/research/graph
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
CUDA_VISIBLE_DEVICES=0 "$PY" -m next_iteration.binding_factorial_score --output outputs/b8_binding_factorial_scores_20260914_v1 --execute
# Contrast evaluation is a separate CPU step after an independent actual-exit-0
# launch receipt binds this completed scorer, model and prediction manifest.
