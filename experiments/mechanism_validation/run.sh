#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../.."
ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
OUT=${OUT:-experiments/mechanism_validation/outputs/ragtruth_v1}
BOOTSTRAP=${BOOTSTRAP:-200}

echo "[1/6] screen train features"
python -m experiments.mechanism_validation.main screen --split-root "$ROOT/train" --output-dir "$OUT/train_features" --device cuda
echo "[2/6] screen test features"
python -m experiments.mechanism_validation.main screen --split-root "$ROOT/test" --output-dir "$OUT/test_features" --device cuda
echo "[3/6] evaluate mechanisms"
python -m experiments.mechanism_validation.main evaluate-mechanisms --train-split "$ROOT/train" --train-features "$OUT/train_features" --test-split "$ROOT/test" --test-features "$OUT/test_features" --output-dir "$OUT/mechanisms" --bootstrap "$BOOTSTRAP"
echo "[4/6] build train graphs"
python -m experiments.mechanism_validation.main build-graph --split-root "$ROOT/train" --mechanism-features "$OUT/train_features" --output-dir "$OUT/train_graphs" --device cuda
echo "[5/6] build test graphs"
python -m experiments.mechanism_validation.main build-graph --split-root "$ROOT/test" --mechanism-features "$OUT/test_features" --output-dir "$OUT/test_graphs" --device cuda
echo "[6/6] evaluate graphs"
python -m experiments.mechanism_validation.main evaluate-graphs --train-split "$ROOT/train" --train-graphs "$OUT/train_graphs" --test-split "$ROOT/test" --test-graphs "$OUT/test_graphs" --output-dir "$OUT/graphs" --bootstrap "$BOOTSTRAP"
