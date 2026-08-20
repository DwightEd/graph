#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-D:/projects/python_projects/research/data/RAGTruth/llama31_8b}
PYTHON=${PYTHON:-python}
BASE_OUT=${BASE_OUT:-experiments/attention_phenomenology/outputs/head_model_ablation_$(date +%Y%m%d_%H%M%S)}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-500}

ROOT="$ROOT" PYTHON="$PYTHON" OUT="$BASE_OUT/reuse" REUSE_TOP_K=5 \
  bash "$SCRIPT_DIR/run_head_model.sh"

ROOT="$ROOT" PYTHON="$PYTHON" OUT="$BASE_OUT/no_reuse" REUSE_TOP_K=5 \
  MASK_RESPONSE_REUSE=1 \
  bash "$SCRIPT_DIR/run_head_model.sh"

"$PYTHON" -u -m experiments.attention_phenomenology.main compare-head-models \
  --reuse-predictions "$BASE_OUT/reuse/test_scores.npz" \
  --no-reuse-predictions "$BASE_OUT/no_reuse/test_scores.npz" \
  --output "$BASE_OUT/comparison.json" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --seed "${SEED:-20260820}"

printf '\nDone: %s\n' "$BASE_OUT/comparison.json"
