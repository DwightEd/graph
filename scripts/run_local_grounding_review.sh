#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PHASE="${1:-pilot}"
OUT="${2:?Supply a NEW absolute output directory}"
if [[ "$PHASE" == validation ]]; then
  echo 'P6 confirmation requires a new frozen source roster; original R04 validation is already used.' >&2
  exit 2
fi
exec "$BASE/conda_envs/research/bin/python" -m next_iteration.local_grounding_review \
  --inputs "$BASE/research/reanchor/outputs/r04_roster_20260913_v1/inputs.jsonl" \
  --model "$BASE/models/Qwen3-8B" --phase "$PHASE" --output "$OUT" --batch-size 4
