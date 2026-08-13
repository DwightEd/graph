#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/data/feature_extraction/lookback_graph/$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$FORMAL_ROOT/train}"
TEST_SPLIT="${TEST_SPLIT:-$FORMAL_ROOT/test}"
DEVICE="${DEVICE:-cuda}"
LAYER_BINS="${LAYER_BINS:-8}"
TSNE_LANDMARKS="${TSNE_LANDMARKS:-10000}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}.matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

printf '[1/5] Reading compressed attention CSR directly (no conversion)\n'
printf 'train_split=%s\ntest_split=%s\noutput=%s\n' "$TRAIN_SPLIT" "$TEST_SPLIT" "$OUTPUT_DIR"

ARGS=(
  discover-patterns
  --train-split "$TRAIN_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --layer-bins "$LAYER_BINS"
  --min-patterns 2
  --max-patterns 6
  --fit-reference-size 30000
  --tsne-landmarks "$TSNE_LANDMARKS"
  --perplexity 40
  --position-bins 10
  --csr-row-block 4096
  --display-mass-cover 0.80
  --display-edges-per-type 2
)

if [[ -n "$SAMPLE_IDS" ]]; then
  IFS=',' read -r -a REQUESTED_SAMPLES <<< "$SAMPLE_IDS"
  for SAMPLE_ID in "${REQUESTED_SAMPLES[@]}"; do
    ARGS+=(--sample-id "$SAMPLE_ID")
  done
fi

"$PYTHON" -u main.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

printf 'complete_output=%s\n' "$OUTPUT_DIR"
printf 'run_log=%s\n' "$LOG_FILE"
printf 'report=%s\n' "$OUTPUT_DIR/lookback_report.json"
printf 'population_tsne=%s\n' "$OUTPUT_DIR/lookback_embedding_tsne.png"
printf 'population_separation=%s\n' "$OUTPUT_DIR/lookback_separation.png"
printf 'sample_graph=%s\n' "$OUTPUT_DIR/sample_*_lookback_graph.png"
