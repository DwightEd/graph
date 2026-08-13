#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/data/feature_extraction/token_representation/$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$FORMAL_ROOT/train}"
TEST_SPLIT="${TEST_SPLIT:-$FORMAL_ROOT/test}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}.matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

printf '[start] compressed CSR -> exact graph scalars -> mass-preserving multi-hop token representations\n'
printf 'train_split=%s\ntest_split=%s\noutput=%s\n' "$TRAIN_SPLIT" "$TEST_SPLIT" "$OUTPUT_DIR"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  printf '[preflight] running graph, data-interface, and end-to-end representation tests\n'
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
fi

ARGS=(
  represent-tokens
  --train-split "$TRAIN_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --position-bins "${POSITION_BINS:-10}"
  --diffusion-hops "${DIFFUSION_HOPS:-2}"
  --csr-row-block "${CSR_ROW_BLOCK:-4096}"
  --display-mass-cover "${DISPLAY_MASS_COVER:-0.80}"
  --display-edges-per-type "${DISPLAY_EDGES_PER_TYPE:-2}"
  --display-max-edges "${DISPLAY_MAX_EDGES:-300}"
  --visual-reference-size "${VISUAL_REFERENCE_SIZE:-30000}"
  --seed "${SEED:-42}"
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
printf 'report=%s\n' "$OUTPUT_DIR/token_representation_report.json"
printf 'embeddings=%s\n' "$OUTPUT_DIR/token_representations_label_free.npz"
printf 'all_sample_graphs=%s\n' "$OUTPUT_DIR/sample_graphs"
printf 'population_figure=%s\n' "$OUTPUT_DIR/population_token_representations.png"
printf 'selected_sample_figure=%s\n' "$OUTPUT_DIR/sample_*_token_graph.png"
