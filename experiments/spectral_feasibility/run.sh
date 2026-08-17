#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
export PYTHONUNBUFFERED=1
BASE_OUT=experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2
if [[ -n "${OUT:-}" ]]; then
  OUT=$OUT
elif [[ -n "${TRAIN_LIMIT:-}" || -n "${TEST_LIMIT:-}" ]]; then
  OUT="$BASE_OUT/smoke_train_${TRAIN_LIMIT:-all}_test_${TEST_LIMIT:-all}"
else
  OUT="$BASE_OUT/full"
fi

FIT_EXTRA=()
SCORE_EXTRA=()
[[ -n "${TRAIN_LIMIT:-}" ]] && FIT_EXTRA+=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && SCORE_EXTRA+=(--limit "$TEST_LIMIT")

if [[ -n "${LIMIT:-}" ]]; then
  echo "LIMIT is no longer accepted: use TRAIN_LIMIT and TEST_LIMIT separately" >&2
  exit 2
fi

mkdir -p "$OUT"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  echo "[preflight] RR spectral contracts"
  "$PYTHON" -m unittest tests.test_spectral_feasibility
fi

echo "dataset_root=$ROOT"
echo "train_split=$ROOT/train"
echo "test_split=$ROOT/test"
echo "output=$OUT"
echo "device=$DEVICE"
echo "top_k=${TOP_K:-5}"
echo "position_bins=${POSITION_BINS:-4}"
echo "pca_dim=${PCA_DIM:-32}"
echo "reference_per_sample=${REFERENCE_PER_SAMPLE:-6}"
echo "trim_fraction=${TRIM_FRACTION:-0.90}"
echo "calibration_fraction=${CALIBRATION_FRACTION:-0.25}"
echo "split_seed=${SPLIT_SEED:-20260815}"
echo "channel_tail_fraction=${CHANNEL_TAIL_FRACTION:-0.05}"
echo "attribution_topk=${ATTRIBUTION_TOPK:-8}"

echo "[1/3] fit label-free RR-only robust subspace on train"
"$PYTHON" -u -m experiments.spectral_feasibility.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --top-k "${TOP_K:-5}" \
  --position-bins "${POSITION_BINS:-4}" \
  --pca-dim "${PCA_DIM:-32}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-6}" \
  --trim-fraction "${TRIM_FRACTION:-0.90}" \
  --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
  --split-seed "${SPLIT_SEED:-20260815}" \
  --channel-tail-fraction "${CHANNEL_TAIL_FRACTION:-0.05}" \
  --attribution-topk "${ATTRIBUTION_TOPK:-8}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  "${FIT_EXTRA[@]}"

echo "[2/3] score test RR spectra without labels"
"$PYTHON" -u -m experiments.spectral_feasibility.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${SCORE_EXTRA[@]}"

echo "[3/3] post-hoc token-label evaluation"
"$PYTHON" -u -m experiments.spectral_feasibility.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device cpu

echo "done: $OUT/evaluation.json"
