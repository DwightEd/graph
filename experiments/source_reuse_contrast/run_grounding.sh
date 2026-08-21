#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-experiments/source_reuse_contrast/outputs/grounding_$RUN_ID}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

LIMIT_TRAIN=()
LIMIT_TEST=()
TASK=()
NO_PROGRESS=()
[[ -n "${TRAIN_LIMIT:-}" ]] && LIMIT_TRAIN=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && LIMIT_TEST=(--limit "$TEST_LIMIT")
[[ -n "${TASK_TYPE:-}" ]] && TASK=(--task-type "$TASK_TYPE")
[[ "${TQDM_DISABLE:-0}" == "1" ]] && NO_PROGRESS=(--no-progress)

mkdir -p "$OUT"

printf '\n[1/3] train label-free grounding-sensitive edge refinement\n'
"$PYTHON" -u -m experiments.source_reuse_contrast.grounding_main train \
  --train-split "$ROOT/train" \
  --output-dir "$OUT/train" \
  --device "$DEVICE" \
  --hidden-dim "${HIDDEN_DIM:-96}" \
  --received-topk "${RECEIVED_TOPK:-5}" \
  --edge-mask-rate "${EDGE_MASK_RATE:-0.25}" \
  --epochs "${EPOCHS:-20}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --score-rounds "${SCORE_ROUNDS:-4}" \
  --seed "${SEED:-20260821}" \
  "${TASK[@]}" "${LIMIT_TRAIN[@]}" "${NO_PROGRESS[@]}"

printf '\n[2/3] freeze reconstruction and counterfactual scores\n'
"$PYTHON" -u -m experiments.source_reuse_contrast.grounding_main score \
  --split-root "$ROOT/test" \
  --checkpoint "$OUT/train/model.pt" \
  --output-dir "$OUT/score" \
  --device "$DEVICE" \
  "${TASK[@]}" "${LIMIT_TEST[@]}"

printf '\n[3/3] unlock labels for post-hoc evaluation\n'
"$PYTHON" -u -m experiments.source_reuse_contrast.grounding_main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/score/scores.npz" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --seed "${SEED:-20260821}"

printf '\nDone: %s\n' "$OUT"
