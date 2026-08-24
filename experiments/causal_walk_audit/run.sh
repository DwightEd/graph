#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the RAGTruth model directory}
OUT=${OUT:-$SCRIPT_DIR/outputs/smoke_$(date -u +%Y%m%dT%H%M%SZ)}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cpu}
TASK_TYPE=${TASK_TYPE:-QA}
TRAIN_LIMIT=${TRAIN_LIMIT:-100}
TEST_LIMIT=${TEST_LIMIT:-30}
ANCHOR_MANIFEST=${ANCHOR_MANIFEST:-}

COMMON=(
  --device "$DEVICE"
  --task-type "$TASK_TYPE"
  --block-rows "${BLOCK_ROWS:-8192}"
  --max-anchors "${MAX_ANCHORS:-12}"
  --prompt-chunk-tokens "${PROMPT_CHUNK_TOKENS:-32}"
  --reservoir-rows "${RESERVOIR_ROWS:-100000}"
  --ridge-alpha "${RIDGE_ALPHA:-1.0}"
  --horizon "${HORIZON:-4}"
  --minimum-anchor-mass "${MINIMUM_ANCHOR_MASS:-0.001}"
  --anchor-shuffle-replicates "${ANCHOR_SHUFFLE_REPLICATES:-8}"
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}"
  --permutation-replicates "${PERMUTATION_REPLICATES:-199}"
  --seed "${SEED:-20260824}"
)
[[ "${TQDM_DISABLE:-0}" == "1" ]] && COMMON+=(--no-progress)
[[ -n "$ANCHOR_MANIFEST" ]] && COMMON+=(--anchor-manifest "$ANCHOR_MANIFEST")

TRAIN_ARGS=()
TEST_ARGS=()
[[ -n "$TRAIN_LIMIT" ]] && TRAIN_ARGS=(--limit "$TRAIN_LIMIT")
[[ -n "$TEST_LIMIT" ]] && TEST_ARGS=(--limit "$TEST_LIMIT")

mkdir -p "$OUT"

"$PYTHON" -u -m experiments.causal_walk_audit.main fit \
  --train-split "$DATA_ROOT/train" \
  --output-dir "$OUT/train" \
  "${COMMON[@]}" "${TRAIN_ARGS[@]}"

"$PYTHON" -u -m experiments.causal_walk_audit.main score \
  --split-root "$DATA_ROOT/test" \
  --model "$OUT/train/model.npz" \
  --output-dir "$OUT/score" \
  "${COMMON[@]}" "${TEST_ARGS[@]}"

"$PYTHON" -u -m experiments.causal_walk_audit.main evaluate \
  --split-root "$DATA_ROOT/test" \
  --score-dir "$OUT/score" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --permutation-replicates "${PERMUTATION_REPLICATES:-199}"

echo "Done: $OUT"
