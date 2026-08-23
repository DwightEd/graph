#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON:-D:/projects/python_projects/.audit_envs/llm_state_lab_py311/Scripts/python.exe}"
DATA_ROOT="${DATA_ROOT:-D:/projects/python_projects/research/data/RAGTruth/llama31_8b}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs/smoke_30}"
TRAIN_LIMIT="${TRAIN_LIMIT-100}"
TEST_LIMIT="${TEST_LIMIT-30}"
DEVICE="${DEVICE:-cpu}"
TASK_TYPE="${TASK_TYPE:-QA}"
BLOCK_ROWS="${BLOCK_ROWS:-4096}"
REFERENCE_CAPACITY="${REFERENCE_CAPACITY:-2048}"
NULL_REPLICATES="${NULL_REPLICATES:-2}"
LAYER_SHUFFLE_REPLICATES="${LAYER_SHUFFLE_REPLICATES:-5}"
SWAP_ROUNDS="${SWAP_ROUNDS:-10}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-50}"
PERMUTATION_REPLICATES="${PERMUTATION_REPLICATES:-49}"
SEED="${SEED:-20260824}"
SCOPE="${SCOPE:-smoke}"
TOKENIZER="${TOKENIZER-}"

if [[ "$SCOPE" == "confirmation" ]]; then
  echo "run.sh does not refit frozen confirmation artifacts; use the confirmation command in README.md" >&2
  exit 2
fi
if [[ "$SCOPE" == "discovery" && -z "$TOKENIZER" ]]; then
  echo "TOKENIZER is required for the content-token discovery outcome" >&2
  exit 2
fi

REFERENCE="$OUTPUT_DIR/reference.npz"
SCORES="$OUTPUT_DIR/scores"
SPLIT_PLAN="$OUTPUT_DIR/split_plan.json"
EVALUATION="$OUTPUT_DIR/evaluation_$SCOPE"
TRAIN_LIMIT_ARGS=()
TEST_LIMIT_ARGS=()
TOKENIZER_ARGS=()
PLAN_ARGS=()

if [[ -n "$TRAIN_LIMIT" ]]; then
  TRAIN_LIMIT_ARGS=(--limit "$TRAIN_LIMIT")
fi
if [[ -n "$TEST_LIMIT" ]]; then
  TEST_LIMIT_ARGS=(--limit "$TEST_LIMIT")
fi
if [[ -n "$TOKENIZER" ]]; then
  TOKENIZER_ARGS=(--tokenizer "$TOKENIZER")
fi
if [[ "$SCOPE" == "discovery" ]]; then
  PLAN_ARGS=(--split-plan "$SPLIT_PLAN")
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m experiments.non_neural_structure_audit.main fit \
  --train-split "$DATA_ROOT/train" \
  --output "$REFERENCE" \
  --device "$DEVICE" \
  --task-type "$TASK_TYPE" \
  --block-rows "$BLOCK_ROWS" \
  --reference-capacity "$REFERENCE_CAPACITY" \
  --seed "$SEED" \
  "${TRAIN_LIMIT_ARGS[@]}"

"$PYTHON_BIN" -m experiments.non_neural_structure_audit.main score \
  --split-root "$DATA_ROOT/test" \
  --reference "$REFERENCE" \
  --output-dir "$SCORES" \
  --device "$DEVICE" \
  --task-type "$TASK_TYPE" \
  --block-rows "$BLOCK_ROWS" \
  --null-replicates "$NULL_REPLICATES" \
  --layer-shuffle-replicates "$LAYER_SHUFFLE_REPLICATES" \
  --swap-rounds "$SWAP_ROUNDS" \
  --seed "$SEED" \
  "${TEST_LIMIT_ARGS[@]}"

if [[ "$SCOPE" == "discovery" ]]; then
  "$PYTHON_BIN" -m experiments.non_neural_structure_audit.main plan \
    --score-dir "$SCORES" \
    --output "$SPLIT_PLAN" \
    --seed "$SEED"
fi

"$PYTHON_BIN" -m experiments.non_neural_structure_audit.main evaluate \
  --split-root "$DATA_ROOT/test" \
  --score-dir "$SCORES" \
  --output-dir "$EVALUATION" \
  --scope "$SCOPE" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --permutation-replicates "$PERMUTATION_REPLICATES" \
  --seed "$SEED" \
  "${PLAN_ARGS[@]}" \
  "${TOKENIZER_ARGS[@]}"

echo "Audit artifacts complete; inspect $EVALUATION/decision_table.csv"
