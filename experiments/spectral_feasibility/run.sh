#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
DEVICE=${DEVICE:-cuda}
BASE_OUT=experiments/spectral_feasibility/outputs/rr_spectral_subspace
if [[ -n "${OUT:-}" ]]; then
  OUT=$OUT
elif [[ -n "${LIMIT:-}" ]]; then
  OUT="$BASE_OUT/smoke_${LIMIT}"
else
  OUT="$BASE_OUT/full"
fi

EXTRA=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA+=(--limit "$LIMIT")
fi

mkdir -p "$OUT"

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
echo "channel_tail_fraction=${CHANNEL_TAIL_FRACTION:-0.05}"
echo "attribution_topk=${ATTRIBUTION_TOPK:-8}"

echo "[1/3] fit label-free RR-only robust subspace on train"
python -m experiments.spectral_feasibility.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --top-k "${TOP_K:-5}" \
  --position-bins "${POSITION_BINS:-4}" \
  --pca-dim "${PCA_DIM:-32}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-6}" \
  --trim-fraction "${TRIM_FRACTION:-0.90}" \
  --channel-tail-fraction "${CHANNEL_TAIL_FRACTION:-0.05}" \
  --attribution-topk "${ATTRIBUTION_TOPK:-8}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  "${EXTRA[@]}"

echo "[2/3] score test RR spectra without labels"
python -m experiments.spectral_feasibility.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/test_scores.npz" \
  --device "$DEVICE" \
  "${EXTRA[@]}"

echo "[3/3] post-hoc token-label evaluation"
python -m experiments.spectral_feasibility.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device cpu

echo "done: $OUT/evaluation.json"
