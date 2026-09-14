#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PHASE="${1:-pilot}"
OUT="${2:?Supply a NEW absolute output directory as second argument}"
exec "$BASE/conda_envs/research/bin/python" -m next_iteration.grounding_contrast \
  --inputs "$BASE/research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl" \
  --model "$BASE/models/Meta-Llama-3.1-8B-Instruct" \
  --output "$OUT" --phase "$PHASE" --batch-size "${BATCH_SIZE:-4}"
