#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

TRAIN_EXTRA=()
TEST_EXTRA=()
if [[ -n "${TRAIN_LIMIT:-}" ]]; then
  TRAIN_EXTRA=(--limit "$TRAIN_LIMIT")
fi
if [[ -n "${TEST_LIMIT:-}" ]]; then
  TEST_EXTRA=(--limit "$TEST_LIMIT")
fi

if [[ -n "${TRAIN_LIMIT:-}" || -n "${TEST_LIMIT:-}" ]]; then
  RUN_NAME="smoke_train${TRAIN_LIMIT:-all}_test${TEST_LIMIT:-all}"
  FIT_BOOTSTRAP_DEFAULT=100
  EVAL_BOOTSTRAP_DEFAULT=100
  RESERVOIR_DEFAULT=1024
  PCA_DEFAULT=16
  MIN_CONDITION_DEFAULT=8
else
  RUN_NAME=full
  FIT_BOOTSTRAP_DEFAULT=500
  EVAL_BOOTSTRAP_DEFAULT=1000
  RESERVOIR_DEFAULT=4096
  PCA_DEFAULT=32
  MIN_CONDITION_DEFAULT=32
fi

OUT=${OUT:-experiments/rr_signal_audit/outputs/$RUN_NAME}

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi
if [[ -e "$OUT/reference.npz" || -e "$OUT/test_scores.npz" || -e "$OUT/evaluation/evaluation.json" ]]; then
  echo "output already contains frozen artifacts; choose a fresh OUT: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT/evaluation"

echo "dataset_root=$ROOT"
echo "output=$OUT"
echo "device=$DEVICE"
echo "top_k=${TOP_K:-5}"
echo "lag_bins=${LAG_BINS:-8}"
echo "reservoir_rows=${RESERVOIR_ROWS:-$RESERVOIR_DEFAULT}"
echo "pca_dim=${PCA_DIM:-$PCA_DEFAULT}"
echo "trim_fraction=${TRIM_FRACTION:-0.90}"

printf '\n[1/3] fit label-free attention signal references\n'
"$PYTHON" -u -m experiments.rr_signal_audit.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --top-k "${TOP_K:-5}" \
  --lag-bins "${LAG_BINS:-8}" \
  --local-lag-max "${LOCAL_LAG_MAX:-4}" \
  --anchor-count "${ANCHOR_COUNT:-8}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --causal-position-bins "${CAUSAL_POSITION_BINS:-10}" \
  --relative-position-bins "${RELATIVE_POSITION_BINS:-4}" \
  --reservoir-rows "${RESERVOIR_ROWS:-$RESERVOIR_DEFAULT}" \
  --pca-dim "${PCA_DIM:-$PCA_DEFAULT}" \
  --min-condition-rows "${MIN_CONDITION_ROWS:-$MIN_CONDITION_DEFAULT}" \
  --trim-fraction "${TRIM_FRACTION:-0.90}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --bootstrap-replicates "${FIT_BOOTSTRAP:-$FIT_BOOTSTRAP_DEFAULT}" \
  --seed "${SEED:-20260818}" \
  "${TRAIN_EXTRA[@]}" || {
    status=$?
    echo "fit failed (exit=$status); score/evaluate were not started" >&2
    exit "$status"
  }

printf '\n[2/3] freeze held-out attention scores without labels\n'
"$PYTHON" -u -m experiments.rr_signal_audit.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${TEST_EXTRA[@]}" || {
    status=$?
    echo "score failed (exit=$status); evaluate was not started" >&2
    exit "$status"
  }

printf '\n[3/3] post-hoc signal metrics and hallucination-onset effects\n'
"$PYTHON" -u -m experiments.rr_signal_audit.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output-dir "$OUT/evaluation" \
  --device cpu \
  --onset-window "${ONSET_WINDOW:-4}" \
  --bootstrap-replicates "${EVAL_BOOTSTRAP:-$EVAL_BOOTSTRAP_DEFAULT}" \
  --seed "${SEED:-20260818}" || {
    status=$?
    echo "evaluate failed (exit=$status)" >&2
    exit "$status"
  }

echo "done: $OUT/evaluation/evaluation.json"
echo "metrics: $OUT/evaluation/score_metrics.csv"
echo "onset:  $OUT/evaluation/onset_effects.csv"
