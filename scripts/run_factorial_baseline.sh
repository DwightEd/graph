#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "usage: $0 FACTORIAL_EVENTS OUTPUT_ROOT [FIT_SPLIT] [SCORE_SPLIT]" >&2
  exit 2
fi

events=${1:-data/pilot_factorial_events.jsonl}
output_root=${2:-outputs/graph_anomaly_pilot}
fit_split=${3:-train}
score_split=${4:-test}

if [[ ! -f "$events" ]]; then
  echo "factorial event JSONL does not exist: $events" >&2
  exit 2
fi

python -m control_graph.cli build \
  --input "$events" \
  --output "$output_root/graphs.jsonl"

python -m control_graph.cli detect \
  --graphs "$output_root/graphs.jsonl" \
  --output "$output_root/detection" \
  --fit-split "$fit_split" \
  --score-split "$score_split"
