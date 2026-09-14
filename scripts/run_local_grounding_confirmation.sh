#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT="${1:?Supply a NEW absolute output directory after the candidate freeze}"
FREEZE="$BASE/research/graph/outputs/P6_CONFIRMATION_FREEZE_20260914.json"
if [[ ! -f "$FREEZE" ]]; then
  echo 'Candidate has not been frozen for confirmation; refusing to score.' >&2
  exit 2
fi
"$BASE/conda_envs/research/bin/python" -m next_iteration.confirmation_freeze_check --freeze "$FREEZE"
exec "$BASE/conda_envs/research/bin/python" -m next_iteration.local_grounding_confirmation \
  --inputs "$BASE/research/graph/outputs/p6_confirmation_roster_20260914_v1/inputs.jsonl" \
  --model "$BASE/models/Qwen3-8B" --phase validation --output "$OUT" --batch-size 4
