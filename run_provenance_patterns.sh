#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/data/feature_extraction/provenance_patterns/$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$FORMAL_ROOT/train}"
TEST_SPLIT="${TEST_SPLIT:-$FORMAL_ROOT/test}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINTS="${CHECKPOINTS:-8}"
TSNE_LANDMARKS="${TSNE_LANDMARKS:-10000}"
SIGNATURE_VIEW="${SIGNATURE_VIEW:-prompt_absorption}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR"

printf '[1/4] Direct sparse-cache input (no attention conversion)\n'
printf 'train_split=%s\ntest_split=%s\n' "$TRAIN_SPLIT" "$TEST_SPLIT"
printf '[2/4] Extracting one provenance curve per response-token node\n'

"$PYTHON" main.py discover-patterns \
  --train-split "$TRAIN_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR/patterns" \
  --device "$DEVICE" \
  --signature-view "$SIGNATURE_VIEW" \
  --selection threshold \
  --checkpoints "$CHECKPOINTS" \
  --min-patterns 2 \
  --max-patterns 6 \
  --fit-reference-size 30000 \
  --tsne-landmarks "$TSNE_LANDMARKS" \
  --perplexity 40 \
  2>&1 | tee "$OUTPUT_DIR/run.log"

printf 'complete_output=%s\n' "$OUTPUT_DIR"
printf 'pattern_report=%s\n' "$OUTPUT_DIR/patterns/pattern_report.json"
printf 'pattern_tsne=%s\n' "$OUTPUT_DIR/patterns/provenance_pattern_tsne.png"
printf 'pattern_curves=%s\n' "$OUTPUT_DIR/patterns/provenance_pattern_curves.png"
printf 'prototype_graphs=%s\n' "$OUTPUT_DIR/patterns/prototype_graphs"
printf 'response_graph_modes=%s\n' "$OUTPUT_DIR/patterns/response_graph_pattern_modes.png"
