#!/usr/bin/env bash
# Test, analyze existing attention caches, then optionally evaluate RAGTruth.
# Activate the research environment before running; PY defaults to its python.
# Relative paths are resolved from the repository root.
#
# Analysis only:
#   CACHE=/path/to/canonical_npz bash experiments/unsupervised_token_graph/run_all.sh
# Analysis + evaluation:
#   CACHE=/path/to/canonical_npz METADATA=/path/to/metadata.jsonl \
#   ANNOTATIONS=/path/to/RAGTruth/response.jsonl \
#   bash experiments/unsupervised_token_graph/run_all.sh
# Extra arguments, e.g. --layers 10 11 --heads 3 7, go to the analysis command.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
CACHE="${CACHE:?Set CACHE to the existing canonical attention NPZ directory}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_v1}"
PATTERN="${PATTERN:-**/*.npz}"
METADATA="${METADATA:-}"
ANNOTATIONS="${ANNOTATIONS:-}"

printf '\n[1/3] Run reanchor tests\n'
"$PY" -u -m pytest \
  tests/test_reanchor_information.py \
  tests/test_reanchor_pipeline.py -q

args=(--cache "$CACHE" --pattern "$PATTERN" --output "$OUTPUT" --resume)
if [[ -n "$METADATA" ]]; then
  args+=(--metadata "$METADATA")
fi

printf '\n[2/3] Analyze attention caches: %s\n' "$CACHE"
# Exact token roots and topology controls stay enabled by default.
# The Python runner checks settings before reusing completed samples.
"$PY" -u -m experiments.unsupervised_token_graph.run "$@" "${args[@]}"

if [[ -n "$ANNOTATIONS" ]]; then
  printf '\n[3/3] Evaluate saved scores\n'
  PY="$PY" OUTPUT="$OUTPUT" ANNOTATIONS="$ANNOTATIONS" \
    bash "$ROOT/experiments/unsupervised_token_graph/evaluate.sh"
else
  printf '\n[3/3] Evaluation skipped: ANNOTATIONS is not set.\n'
fi

printf '\nFinished: %s\n' "$OUTPUT"
