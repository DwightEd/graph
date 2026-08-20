#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
OUT=${OUT:-experiments/attention_phenomenology/outputs/full}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

FIT_LIMIT=()
SCORE_LIMIT=()
[[ -n "${TRAIN_LIMIT:-}" ]] && FIT_LIMIT=(--limit "$TRAIN_LIMIT")
[[ -n "${TEST_LIMIT:-}" ]] && SCORE_LIMIT=(--limit "$TEST_LIMIT")

CONFIG=(
  --null-prompt-position-bins "${NULL_PROMPT_POSITION_BINS:-4}"
  --null-response-lag-bins "${NULL_RESPONSE_LAG_BINS:-8}"
  --recent-response-tokens "${RECENT_RESPONSE_TOKENS:-4}"
  --causal-position-bins "${CAUSAL_POSITION_BINS:-10}"
  --reference-minimum-scale "${REFERENCE_MINIMUM_SCALE:-0.001}"
  --maximum-standardized-value "${MAXIMUM_STANDARDIZED_VALUE:-10}"
  --block-rows "${BLOCK_ROWS:-8192}"
  --seed "${SEED:-20260819}"
)

mkdir -p "$OUT"

printf '\n[1/3] fit unlabeled attention-routing reference\n'
"$PYTHON" -u -m experiments.attention_phenomenology.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --reservoir-rows "${RESERVOIR_ROWS:-2048}" \
  "${CONFIG[@]}" \
  "${FIT_LIMIT[@]}"

DETAIL_ARGS=()
if [[ -n "${DETAIL_SAMPLE_IDS:-}" ]]; then
  IFS=',' read -ra IDS <<< "$DETAIL_SAMPLE_IDS"
  for ID in "${IDS[@]}"; do
    DETAIL_ARGS+=(--detail-sample-id "$ID")
  done
fi

printf '\n[2/3] freeze mechanism fields and exact-endpoint null\n'
"$PYTHON" -u -m experiments.attention_phenomenology.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output-dir "$OUT/scores" \
  --device "$DEVICE" \
  "${CONFIG[@]}" \
  "${DETAIL_ARGS[@]}" \
  "${SCORE_LIMIT[@]}"

printf '\n[3/3] unlock labels for onset, lock-in, and topology tests\n'
"$PYTHON" -u -m experiments.attention_phenomenology.main evaluate \
  --split-root "$ROOT/test" \
  --score-dir "$OUT/scores" \
  --output-dir "$OUT/evaluation" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED:-20260819}"

printf '\nDone: %s\n' "$OUT/evaluation/evaluation.json"
