#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 PREPARED_RESPONSES LOCAL_LLAMA OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi

input=$1
model=$2
output=$3
device=${4:-cpu}

if [[ ! -f "$input" || ! -d "$model" || -e "$output" ]]; then
  echo "input and local model must exist; output must be new" >&2
  exit 2
fi

python main.py extract --input "$input" --model "$model" \
  --output "$output/features" --device "$device"
python main.py detect --features "$output/features" --output "$output/detection"
