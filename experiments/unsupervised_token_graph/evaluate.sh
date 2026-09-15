#!/usr/bin/env bash
# Evaluate already-saved measurements without recomputing graphs.
# Use the same OUTPUT as run_all.sh. No CACHE path is needed here.
#
# OUTPUT=outputs/source_carrier_information_v1 \
# ANNOTATIONS=/path/to/RAGTruth/response.jsonl \
# bash experiments/unsupervised_token_graph/evaluate.sh
# Optional: PY=/path/to/python SPLIT=test BOOTSTRAP=200 CHANNEL_QUANTILE=0.9.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_v1}"
ANNOTATIONS="${ANNOTATIONS:?Set ANNOTATIONS to the RAGTruth response.jsonl file}"

"$PY" -u -m experiments.unsupervised_token_graph.run evaluate \
  --predictions "$OUTPUT" \
  --annotations "$ANNOTATIONS" \
  --output "$OUTPUT/evaluation.json" \
  --split "${SPLIT:-test}" \
  --bootstrap "${BOOTSTRAP:-200}" \
  --channel-quantile "${CHANNEL_QUANTILE:-0.9}" \
  "$@"

printf '\nEvaluation saved: %s/evaluation.json\n' "$OUTPUT"
