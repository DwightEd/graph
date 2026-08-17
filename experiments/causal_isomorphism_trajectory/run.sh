#!/usr/bin/env bash
set -euo pipefail

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
  BOOTSTRAP_DEFAULT=100
else
  RUN_NAME=full
  BOOTSTRAP_DEFAULT=1000
fi
OUT=${OUT:-experiments/causal_isomorphism_trajectory/outputs/v1/$RUN_NAME}

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi

mkdir -p "$OUT"

echo "dataset_root=$ROOT"
echo "output=$OUT"
echo "device=$DEVICE"
echo "layer_bands=${LAYER_BANDS:-8}"
echo "hash_dim=${HASH_DIM:-128}"
echo "pca_dim=${PCA_DIM:-32}"
echo "reference_per_sample=${REFERENCE_PER_SAMPLE:-16}"

printf '\n[1/3] fit label-free causal isomorphism trajectory geometry\n'
"$PYTHON" -u -m experiments.causal_isomorphism_trajectory.main fit \
  --train-split "$ROOT/train" \
  --output-dir "$OUT" \
  --device "$DEVICE" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --layer-bands "${LAYER_BANDS:-8}" \
  --max-rp-events-per-band "${MAX_RP_EVENTS_PER_BAND:-2}" \
  --max-rr-events-per-band "${MAX_RR_EVENTS_PER_BAND:-4}" \
  --hash-dim "${HASH_DIM:-128}" \
  --lag-bins "${LAG_BINS:-8}" \
  --weight-bins "${WEIGHT_BINS:-5}" \
  --position-buckets "${POSITION_BUCKETS:-10}" \
  --late-band-transitions "${LATE_BAND_TRANSITIONS:-2}" \
  --source-anchor-count "${SOURCE_ANCHOR_COUNT:-8}" \
  --max-parent-events "${MAX_PARENT_EVENTS:-8}" \
  --pca-dim "${PCA_DIM:-32}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-16}" \
  --min-condition-rows "${MIN_CONDITION_ROWS:-32}" \
  --trim-fraction "${TRIM_FRACTION:-1.0}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --topology-gate-min-coverage "${TOPOLOGY_GATE_MIN_COVERAGE:-0.25}" \
  --seed "${SEED:-20260817}" \
  "${TRAIN_EXTRA[@]}"

printf '\n[2/3] freeze held-out test trajectory scores without labels\n'
"$PYTHON" -u -m experiments.causal_isomorphism_trajectory.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${TEST_EXTRA[@]}"

printf '\n[3/3] post-hoc token-label evaluation\n'
"$PYTHON" -u -m experiments.causal_isomorphism_trajectory.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device cpu

echo "done: $OUT/evaluation.json"
