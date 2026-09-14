#!/usr/bin/env bash
set -euo pipefail
BASE=/share/home/tm902089733300000/a903202310/lys
cd "$BASE/research/graph"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT="${1:?Supply a NEW absolute output directory after the P7 confirmation freeze}"
FREEZE="$BASE/research/graph/outputs/P7_CONFIRMATION_FREEZE_20260914.json"
if [[ ! -f "$FREEZE" ]]; then
  echo 'P7 candidate is not frozen; refusing confirmation scoring.' >&2
  exit 2
fi
"$BASE/conda_envs/research/bin/python" -m next_iteration.reasoned_confirmation_freeze_check --freeze "$FREEZE"
exec "$BASE/conda_envs/research/bin/python" -m next_iteration.reasoned_confirmation \
  --inputs "$BASE/research/graph/outputs/p7_confirmation_roster_20260914_v1/inputs.jsonl" \
  --model "$BASE/models/Qwen3-8B" --phase validation --output "$OUT" --batch-size 4
