#!/usr/bin/env bash
# Evaluate saved scores only. Default: the test output from run_all.sh.
# SPLIT=train selects train. --completed-only previews an interrupted run.
# Preview reports use evaluation_partial.json; scoring/checkpoint files are unchanged.
# ANNOTATIONS overrides saved settings and automatic cache-ancestor discovery.
# The evaluator finds an existing RAGTruth/response.jsonl without rescoring.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_npz_v2/$SPLIT}"
ANNOTATIONS="${ANNOTATIONS:-}"
COMPLETED_ONLY="${COMPLETED_ONLY:-0}"
for arg in "$@"; do
  if [[ "$arg" == "--completed-only" ]]; then COMPLETED_ONLY=1; fi
done

report="$OUTPUT/evaluation.json"
if [[ "$COMPLETED_ONLY" == 1 ]]; then report="$OUTPUT/evaluation_partial.json"; fi
args=(--predictions "$OUTPUT" --output "$report"
      --split "$SPLIT" --bootstrap "${BOOTSTRAP:-200}"
      --channel-quantile "${CHANNEL_QUANTILE:-0.9}")
if [[ "$COMPLETED_ONLY" == 1 ]]; then args+=(--completed-only); fi
if [[ -n "$ANNOTATIONS" ]]; then args+=(--annotations "$ANNOTATIONS"); fi

"$PY" -u -m experiments.unsupervised_token_graph.run evaluate "${args[@]}" "$@"
