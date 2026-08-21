#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-experiments/source_reuse_contrast/outputs/$RUN_ID}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
MODES=${MODES:-current,birth,dynamic}

TRAIN_LIMIT_ARGS=()
TEST_LIMIT_ARGS=()
PROGRESS_ARGS=()
[[ -n "${TRAIN_LIMIT:-}" ]] && TRAIN_LIMIT_ARGS=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && TEST_LIMIT_ARGS=(--limit "$TEST_LIMIT")
[[ "${NO_PROGRESS:-0}" == "1" ]] && PROGRESS_ARGS=(--no-progress)

mkdir -p "$OUT"
IFS=',' read -r -a MODE_ARRAY <<< "$MODES"
SCORE_ARGS=()
TRAINING_ARGS=()
MANIFEST_ARGS=()

for MODE in "${MODE_ARRAY[@]}"; do
  printf '\n[%s] train masked-source predictor without labels\n' "$MODE"
  "$PYTHON" -u -m experiments.source_reuse_contrast.main train \
    --train-split "$ROOT/train" \
    --output-dir "$OUT/$MODE/train" \
    --device "$DEVICE" \
    --memory-mode "$MODE" \
    --hidden-dim "${HIDDEN_DIM:-64}" \
    --temperature "${TEMPERATURE:-0.2}" \
    --negative-count "${NEGATIVE_COUNT:-4}" \
    --negative-pool-size "${NEGATIVE_POOL_SIZE:-32}" \
    --epochs "${EPOCHS:-20}" \
    --learning-rate "${LEARNING_RATE:-0.001}" \
    --weight-decay "${WEIGHT_DECAY:-0.00001}" \
    --validation-fraction "${VALIDATION_FRACTION:-0.1}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-3}" \
    --score-rounds "${SCORE_ROUNDS:-4}" \
    --seed "${SEED:-20260820}" \
    "${PROGRESS_ARGS[@]}" \
    "${TRAIN_LIMIT_ARGS[@]}"

  printf '\n[%s] freeze raw NLL, margins, controls, and embeddings\n' "$MODE"
  "$PYTHON" -u -m experiments.source_reuse_contrast.main score \
    --split-root "$ROOT/test" \
    --checkpoint "$OUT/$MODE/train/model.pt" \
    --output-dir "$OUT/$MODE/score" \
    --device "$DEVICE" \
    "${TEST_LIMIT_ARGS[@]}"
  SCORE_ARGS+=(--score "$MODE=$OUT/$MODE/score/scores.npz")
  TRAINING_ARGS+=(--training "$MODE=$OUT/$MODE/train/training.json")
  MANIFEST_ARGS+=(--manifest "$MODE=$OUT/$MODE/score/manifest.json")
done

printf '\n[gate] compare predictive NLL before labels are opened\n'
"$PYTHON" -u -m experiments.source_reuse_contrast.main gate \
  "${TRAINING_ARGS[@]}" \
  "${MANIFEST_ARGS[@]}" \
  --output "$OUT/predictability_gate.json"

printf '\n[evaluate] unlock labels only after every score is frozen\n'
"$PYTHON" -u -m experiments.source_reuse_contrast.main evaluate \
  --split-root "$ROOT/test" \
  "${SCORE_ARGS[@]}" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --seed "${SEED:-20260820}"

printf '\nDone: %s\n' "$OUT"
