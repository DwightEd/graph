#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$BASE/conda_envs/research/bin/python" -m next_iteration.evaluate_p6_frozen_confirmation --execute
