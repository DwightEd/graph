#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 FACTORIAL_EVENTS OUTPUT_ROOT [FIT_SPLIT] [SCORE_SPLIT]" >&2
  exit 2
fi

events=$1
output_root=$2
fit_split=${3:-train}
score_split=${4:-test}

if [[ ! -f "$events" ]]; then
  echo "factorial event JSONL does not exist: $events" >&2
  exit 2
fi

python main.py build \
  --input "$events" \
  --output "$output_root/graphs.jsonl"

python main.py detect \
  --graphs "$output_root/graphs.jsonl" \
  --output "$output_root/detection" \
  --fit-split "$fit_split" \
  --score-split "$score_split"
