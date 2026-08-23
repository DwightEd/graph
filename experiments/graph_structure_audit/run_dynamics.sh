#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:?Set ROOT to the attention-cache directory}
OUT=${OUT:-experiments/graph_structure_audit/outputs/dynamics_$(date -u +%Y%m%dT%H%M%SZ)}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

TASK=()
TRAIN_LIMIT_ARG=()
TEST_LIMIT_ARG=()
NO_PROGRESS=()
[[ -n "${TASK_TYPE:-}" ]] && TASK=(--task-type "$TASK_TYPE")
[[ -n "${TRAIN_LIMIT:-}" ]] && TRAIN_LIMIT_ARG=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && TEST_LIMIT_ARG=(--limit "$TEST_LIMIT")
[[ "${TQDM_DISABLE:-0}" == "1" ]] && NO_PROGRESS=(--no-progress)

mkdir -p "$OUT"

"$PYTHON" -u -m experiments.graph_structure_audit.dynamics_main train \
  --train-split "$ROOT/train" \
  --output-dir "$OUT/train" \
  --device "$DEVICE" \
  --hidden-dim "${HIDDEN_DIM:-96}" \
  --input-dropout "${INPUT_DROPOUT:-0.1}" \
  --epochs "${EPOCHS:-15}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --score-rounds "${SCORE_ROUNDS:-3}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --seed "${SEED:-20260823}" \
  "${TASK[@]}" "${TRAIN_LIMIT_ARG[@]}" "${NO_PROGRESS[@]}"

"$PYTHON" -u -m experiments.graph_structure_audit.dynamics_main score \
  --split-root "$ROOT/test" \
  --checkpoint "$OUT/train/model.pt" \
  --output-dir "$OUT/score" \
  --device "$DEVICE" \
  "${TASK[@]}" "${TEST_LIMIT_ARG[@]}"

"$PYTHON" -u -m experiments.graph_structure_audit.dynamics_main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/score/scores.npz" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED:-20260823}"

echo "Done: $OUT"
