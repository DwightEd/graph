#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
DATA_ROOT="${DATA_ROOT:-$BASE/data/RAGTruth/model_traces/llama31_8b}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/token_representation/$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$DATA_ROOT/train}"
TEST_SPLIT="${TEST_SPLIT:-$DATA_ROOT/test}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_IDS="${SAMPLE_IDS:-}"
DISPLAY_LAYER="${DISPLAY_LAYER:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}.matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

printf '[start] compressed CSR -> 1024-D Lookback -> exact-channel one-hop evidence-flow graph\n'
printf 'train_split=%s\ntest_split=%s\noutput=%s\n' "$TRAIN_SPLIT" "$TEST_SPLIT" "$OUTPUT_DIR"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  printf '[preflight] running Lookback graph representation tests\n'
  "$PYTHON" -m pytest -q \
    tests/test_attention_graph.py \
    tests/test_data.py \
    tests/test_evidence_flow.py \
    tests/test_token_representation.py
fi

ARGS=(
  represent-tokens
  --train-split "$TRAIN_SPLIT"
  --test-split "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
  --position-bins "${POSITION_BINS:-10}"
  --provenance-hops "${PROVENANCE_HOPS:-2}"
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-200}"
  --csr-row-block "${CSR_ROW_BLOCK:-65536}"
  --display-mass-cover "${DISPLAY_MASS_COVER:-0.80}"
  --display-edges-per-type "${DISPLAY_EDGES_PER_TYPE:-2}"
  --display-max-edges "${DISPLAY_MAX_EDGES:-300}"
  --reference-size "${REFERENCE_SIZE:-12000}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-50}"
  --subspace-components "${SUBSPACE_COMPONENTS:-32}"
  --tail-fraction "${TAIL_FRACTION:-0.05}"
  --anomaly-quantile "${ANOMALY_QUANTILE:-0.95}"
  --seed "${SEED:-42}"
)

if [[ -n "$SAMPLE_IDS" ]]; then
  IFS=',' read -r -a REQUESTED_SAMPLES <<< "$SAMPLE_IDS"
  for SAMPLE_ID in "${REQUESTED_SAMPLES[@]}"; do
    ARGS+=(--sample-id "$SAMPLE_ID")
  done
fi

if [[ -n "$DISPLAY_LAYER" ]]; then
  ARGS+=(--display-layer "$DISPLAY_LAYER")
fi

"$PYTHON" -u main.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

if [[ -n "$SAMPLE_IDS" ]]; then
  for SAMPLE_ID in "${REQUESTED_SAMPLES[@]}"; do
    RENDER_ARGS=(
      render-token-graph
      --test-split "$TEST_SPLIT"
      --output-dir "$OUTPUT_DIR"
      --sample-id "$SAMPLE_ID"
      --device "$DEVICE"
    )
    if [[ -n "$DISPLAY_LAYER" ]]; then
      RENDER_ARGS+=(--display-layer "$DISPLAY_LAYER")
    fi
    "$PYTHON" -u main.py "${RENDER_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  done
fi

printf 'complete_output=%s\n' "$OUTPUT_DIR"
printf 'run_log=%s\n' "$LOG_FILE"
printf 'report=%s\n' "$OUTPUT_DIR/token_representation_report.json"
printf 'X=%s\n' "$OUTPUT_DIR/token_node_representations.float16.npy"
printf 'Z=%s\n' "$OUTPUT_DIR/true_graph_node_representations.float16.npy"
