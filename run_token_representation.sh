#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
DATA_ROOT="${DATA_ROOT:-$BASE/data/RAGTruth/model_traces/llama31_8b}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$DATA_ROOT/train}"
TEST_SPLIT="${TEST_SPLIT:-$DATA_ROOT/test}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/causal_topology/$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
DEVICE="${DEVICE:-cuda}"

# REFERENCE_SIZE is the total fit + calibration token budget. The experiment
# assigns source/sample groups to disjoint halves before bottom-k sampling.
REFERENCE_SIZE="${REFERENCE_SIZE:-12000}"
FOURIER_FREQUENCIES="${FOURIER_FREQUENCIES:-4}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}.matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

exec > >(tee -a "$LOG_FILE") 2>&1
printf 'Label-free causal attention topology experiment\n'
printf 'train=%s\ntest=%s\noutput=%s\n' "$TRAIN_SPLIT" "$TEST_SPLIT" "$OUTPUT_DIR"
printf 'reference_budget=%s total fit+cal tokens\nfourier_frequencies=%s\n' \
  "$REFERENCE_SIZE" "$FOURIER_FREQUENCIES"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  printf '\n[preflight] causal topology contracts\n'
  "$PYTHON" -m pytest -q \
    tests/test_causal_topology.py \
    tests/test_one_class.py \
    tests/test_aligned_reservoir.py \
    tests/test_topology_one_class.py \
    tests/test_topology_experiment.py
fi

printf '\n[run] encode topology, fit independent one-class references, score test tokens\n'
"$PYTHON" -u main.py represent-tokens \
  --train-split "$TRAIN_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --position-bins "${POSITION_BINS:-10}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-200}" \
  --reference-size "$REFERENCE_SIZE" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-250}" \
  --subspace-components "${SUBSPACE_COMPONENTS:-32}" \
  --tail-fraction "${TAIL_FRACTION:-0.05}" \
  --fourier-frequencies "$FOURIER_FREQUENCIES" \
  --row-block-size "${ROW_BLOCK_SIZE:-4096}" \
  --seed "${SEED:-42}"

printf '\ncomplete\n'
printf 'report=%s\n' "$OUTPUT_DIR/topology_experiment_report.json"
printf 'label_free_scores=%s\n' "$OUTPUT_DIR/topology_label_free.npz"
printf 'label_free_model=%s\n' "$OUTPUT_DIR/topology_one_class_model.npz"
printf 'log=%s\n' "$LOG_FILE"
