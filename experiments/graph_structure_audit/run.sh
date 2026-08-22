#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:?set ROOT to the attention cache root}
OUT=${OUT:-experiments/graph_structure_audit/outputs/run_$(date -u +%Y%m%dT%H%M%SZ)}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
TASK_ARGS=()
TRAIN_LIMIT_ARGS=()
TEST_LIMIT_ARGS=()
PROGRESS_ARGS=()

[[ -n "${TASK_TYPE:-}" ]] && TASK_ARGS=(--task-type "$TASK_TYPE")
[[ -n "${TRAIN_LIMIT:-}" ]] && TRAIN_LIMIT_ARGS=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && TEST_LIMIT_ARGS=(--limit "$TEST_LIMIT")
[[ "${TQDM_DISABLE:-0}" == "1" ]] && PROGRESS_ARGS=(--no-progress)

mkdir -p "$OUT"

"$PYTHON" -u -m experiments.graph_structure_audit.main train \
  --train-split "$ROOT/train" \
  --output-dir "$OUT/train" \
  --device "$DEVICE" \
  --representation "${REPRESENTATION:-full}" \
  --hidden-dim "${HIDDEN_DIM:-96}" \
  --epochs "${EPOCHS:-15}" \
  --score-rounds "${SCORE_ROUNDS:-4}" \
  --seed "${SEED:-20260822}" \
  "${TASK_ARGS[@]}" "${TRAIN_LIMIT_ARGS[@]}" "${PROGRESS_ARGS[@]}"

"$PYTHON" -u -m experiments.graph_structure_audit.main score \
  --split-root "$ROOT/test" \
  --checkpoint "$OUT/train/model.pt" \
  --output-dir "$OUT/score" \
  --device "$DEVICE" \
  "${TASK_ARGS[@]}" "${TEST_LIMIT_ARGS[@]}"

"$PYTHON" -u -m experiments.graph_structure_audit.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/score/scores.npz" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED:-20260822}"

printf 'Done: %s\n' "$OUT"
