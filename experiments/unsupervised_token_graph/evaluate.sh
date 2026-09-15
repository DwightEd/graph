#!/usr/bin/env bash
# Evaluate saved scores only. Default: the test output from run_all.sh.
# SPLIT=train selects train. --completed-only previews an interrupted run.
# Preview reports use evaluation_partial.json; scoring/checkpoint files are unchanged.
# Default labels: the confirmed RAGTruth/dataset/response.jsonl on this server.
# Missing splits are checked against input cache folders and official annotations.
# TOKENIZER is optional: verify missing offsets with the original local tokenizer.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_npz_v2/$SPLIT}"
ANNOTATIONS="${ANNOTATIONS:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl}"
TOKENIZER="${TOKENIZER:-}"
SOURCE_INFO="${SOURCE_INFO:-}"
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
if [[ -n "$TOKENIZER" ]]; then args+=(--tokenizer "$TOKENIZER"); fi
if [[ -n "$SOURCE_INFO" ]]; then args+=(--source-info "$SOURCE_INFO"); fi

"$PY" -u -m experiments.unsupervised_token_graph.run evaluate "${args[@]}" "$@"
