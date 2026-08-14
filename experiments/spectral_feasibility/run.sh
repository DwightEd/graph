#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
OUT=${OUT:-experiments/spectral_feasibility/outputs/ragtruth_v1}
DEVICE=${DEVICE:-cpu}

EXTRA_EXTRACT=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_EXTRACT+=(--limit "$LIMIT")
fi

mkdir -p "$OUT"

echo "dataset_root=$ROOT"
echo "train_split=$ROOT/train"
echo "test_split=$ROOT/test"
echo "output=$OUT"
echo "device=$DEVICE"

echo "[1/4] extract train spectral representations"
python -m experiments.spectral_feasibility.main extract \
  --split-root "$ROOT/train" \
  --output "$OUT/train_features.npz" \
  --device "$DEVICE" \
  "${EXTRA_EXTRACT[@]}"

echo "[2/4] extract test spectral representations"
python -m experiments.spectral_feasibility.main extract \
  --split-root "$ROOT/test" \
  --output "$OUT/test_features.npz" \
  --device "$DEVICE" \
  "${EXTRA_EXTRACT[@]}"

echo "[3/4] fit unlabeled train reference and score test"
python -m experiments.spectral_feasibility.main score \
  --train-features "$OUT/train_features.npz" \
  --test-features "$OUT/test_features.npz" \
  --output "$OUT/test_scores.npz"

echo "[4/4] post-hoc label evaluation"
python -m experiments.spectral_feasibility.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/test_scores.npz" \
  --output "$OUT/evaluation.json" \
  --device "$DEVICE"

echo "done: $OUT/evaluation.json"
