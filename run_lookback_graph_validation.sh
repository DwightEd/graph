#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"
PYTHON="${PYTHON:-$BASE/conda_envs/research/bin/python}"
ROOT="${ROOT:-$BASE/data/RAGTruth/model_traces/llama31_8b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/lookback_graph_validation/$(date -u +%Y%m%dT%H%M%SZ)}"
DEVICE="${DEVICE:-cuda}"
BOOTSTRAP="${BOOTSTRAP:-200}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-100000}"

mkdir -p "$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/.matplotlib}"
exec > >(tee -a "$OUTPUT_ROOT/run.log") 2>&1

printf 'Lookback graph revalidation\n'
printf 'train=%s\ntest=%s\noutput=%s\n' "$ROOT/train" "$ROOT/test" "$OUTPUT_ROOT"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  "$PYTHON" -m pytest -q \
    experiments/mechanism_validation/tests/test_mechanisms.py \
    experiments/mechanism_validation/tests/test_experiment.py \
    tests/test_attention_graph.py \
    tests/test_data.py \
    tests/test_evidence_flow.py \
    tests/test_token_representation.py
fi

printf '\n[1/4] label-free mechanism screen: train\n'
if [[ -f "$OUTPUT_ROOT/mechanisms/train_features/metadata.json" && \
      -f "$OUTPUT_ROOT/mechanisms/train_features/index.json" ]]; then
  printf 'reusing completed train mechanism features\n'
else
  "$PYTHON" -u -m experiments.mechanism_validation.main screen \
    --split-root "$ROOT/train" \
    --output-dir "$OUTPUT_ROOT/mechanisms/train_features" \
    --device "$DEVICE"
fi

printf '\n[2/4] label-free mechanism screen: test\n'
if [[ -f "$OUTPUT_ROOT/mechanisms/test_features/metadata.json" && \
      -f "$OUTPUT_ROOT/mechanisms/test_features/index.json" ]]; then
  printf 'reusing completed test mechanism features\n'
else
  "$PYTHON" -u -m experiments.mechanism_validation.main screen \
    --split-root "$ROOT/test" \
    --output-dir "$OUTPUT_ROOT/mechanisms/test_features" \
    --device "$DEVICE"
fi

printf '\n[3/4] freeze exact 1024-D Lookback graph representation before labels\n'
"$PYTHON" -u main.py represent-tokens \
  --train-split "$ROOT/train" \
  --test-split "$ROOT/test" \
  --output-dir "$OUTPUT_ROOT/graph_representation" \
  --device "$DEVICE" \
  --bootstrap-replicates "$BOOTSTRAP" \
  --csr-row-block "${CSR_ROW_BLOCK:-65536}" \
  --position-bins "${POSITION_BINS:-10}" \
  --provenance-hops "${PROVENANCE_HOPS:-2}" \
  --reference-size "${REFERENCE_SIZE:-12000}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-50}" \
  --subspace-components "${SUBSPACE_COMPONENTS:-32}" \
  --tail-fraction "${TAIL_FRACTION:-0.05}" \
  --anomaly-quantile "${ANOMALY_QUANTILE:-0.95}" \
  --seed "${SEED:-42}"

printf '\n[4/4] post-hoc supervised scalar Lookback-ratio diagnostic\n'
"$PYTHON" -u -m experiments.mechanism_validation.main evaluate-lookback \
  --train-split "$ROOT/train" \
  --train-features "$OUTPUT_ROOT/mechanisms/train_features" \
  --test-split "$ROOT/test" \
  --test-features "$OUTPUT_ROOT/mechanisms/test_features" \
  --output-dir "$OUTPUT_ROOT/lookback_diagnostic" \
  --bootstrap "$BOOTSTRAP" \
  --max-train-tokens "$MAX_TRAIN_TOKENS"

printf '\ncomplete\n'
printf 'lookback_result=%s\n' "$OUTPUT_ROOT/lookback_diagnostic/results.json"
printf 'graph_result=%s\n' "$OUTPUT_ROOT/graph_representation/token_representation_report.json"
printf 'log=%s\n' "$OUTPUT_ROOT/run.log"
