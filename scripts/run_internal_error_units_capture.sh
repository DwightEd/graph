#!/usr/bin/env bash
set -euo pipefail
cd /share/home/tm902089733300000/a903202310/lys/research/graph
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
phase="${1:?pilot or full}"; version="${2:?fresh version}"
exec /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.internal_error_units_capture \
  --inputs outputs/n9_internal_error_units_inputs_20260914_v3/inputs.jsonl \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --phase "$phase" --output "outputs/n9_internal_error_units_${phase}_20260914_${version}" --max-seconds 5400
