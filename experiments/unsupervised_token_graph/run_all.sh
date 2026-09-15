#!/usr/bin/env bash
# Foreground: test -> analyze existing NPZs -> evaluate when annotations are available.
# Native NPZ identity fields and nearby inputs/records indexes are reused.
# Optional POPULATION points to an EXISTING population; no new METADATA is needed.
# Relative paths are resolved from the repository root.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
CACHE="${CACHE:?Set CACHE to the existing attention NPZ file or directory}"
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
  tests/test_npz_inputs.py -q

args=(--cache "$CACHE" --pattern "$PATTERN" --output "$OUTPUT" --resume)
if [[ -n "$POPULATION" ]]; then args+=(--population "$POPULATION"); fi
if [[ -n "$INDEX" ]]; then args+=(--index "$INDEX"); fi
if [[ -n "$METADATA" ]]; then args+=(--metadata "$METADATA"); fi

printf '\n[2/3] Analyze existing attention: %s\n' "$CACHE"
"$PY" -u -m experiments.unsupervised_token_graph.run "$@" "${args[@]}"

printf '\n[3/3] Evaluate saved scores when a RAGTruth path is available\n'
PY="$PY" OUTPUT="$OUTPUT" ANNOTATIONS="$ANNOTATIONS" \
  bash "$ROOT/experiments/unsupervised_token_graph/evaluate.sh" --if-available

printf '\nFinished: %s\n' "$OUTPUT"
