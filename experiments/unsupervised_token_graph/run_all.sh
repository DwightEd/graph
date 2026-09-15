#!/usr/bin/env bash
# Foreground: test once, then analyze/evaluate train and test separately.
# Defaults are the existing llama31_8b cache directories on the shared server.
# SPLIT=train or SPLIT=test selects one; explicit CACHE retains single-cache mode.
# NPZ identities and nearby indexes are reused. POPULATION/INDEX remain optional.
# Relative paths are resolved from the repository root.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
ATTENTION_ROOT="${ATTENTION_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b}"
TRAIN_CACHE="${TRAIN_CACHE:-$ATTENTION_ROOT/train}"
TEST_CACHE="${TEST_CACHE:-$ATTENTION_ROOT/test}"
CACHE="${CACHE:-}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_npz_v2}"
PATTERN="${PATTERN:-**/*.npz}"
POPULATION="${POPULATION:-}"
INDEX="${INDEX:-}"
METADATA="${METADATA:-}"  # Backward compatibility only.
ANNOTATIONS="${ANNOTATIONS:-}"

printf '\n[1/3] Run reanchor and NPZ interface tests\n'
"$PY" -u -m pytest \
  tests/test_reanchor_information.py \
  tests/test_reanchor_pipeline.py \
  tests/test_npz_inputs.py \
  tests/test_attention_cache_dtypes.py -q

run_split() {
  local split="$1" cache="$2" output="$3"
  shift 3
  local args=(--cache "$cache" --pattern "$PATTERN" --output "$output" --resume)
  if [[ -n "$POPULATION" ]]; then args+=(--population "$POPULATION"); fi
  if [[ -n "$INDEX" ]]; then args+=(--index "$INDEX"); fi
  if [[ -n "$METADATA" ]]; then args+=(--metadata "$METADATA"); fi

  printf '\n[2/3][%s] CACHE=%s\nOUTPUT=%s\n' "$split" "$cache" "$output"
  "$PY" -u -m experiments.unsupervised_token_graph.run "$@" "${args[@]}"

  printf '\n[3/3][%s] Evaluate saved scores when annotations are configured\n' "$split"
  PY="$PY" OUTPUT="$output" SPLIT="$split" ANNOTATIONS="$ANNOTATIONS" \
    bash "$ROOT/experiments/unsupervised_token_graph/evaluate.sh" --if-available
}

if [[ -n "$CACHE" ]]; then
  # Explicit CACHE keeps the previous meaning: OUTPUT is the exact result folder.
  run_split "${SPLIT:-test}" "$CACHE" "$OUTPUT" "$@"
else
  read -r -a splits <<< "${SPLIT:-train test}"
  for split in "${splits[@]}"; do
    case "$split" in
      train) cache="$TRAIN_CACHE" ;;
      test) cache="$TEST_CACHE" ;;
      *) printf 'SPLIT must be train or test, or "train test".\n' >&2; exit 2 ;;
    esac
    run_split "$split" "$cache" "$OUTPUT/$split" "$@"
  done
fi

printf '\nFinished: %s\n' "$OUTPUT"
