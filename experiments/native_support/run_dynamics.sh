#!/usr/bin/env bash
# Usage: bash experiments/native_support/run_dynamics.sh DATA MODEL OUTPUT EXCLUDE [TRAIN=64] [TEST=32]
# EXCLUDE is an already inspected support directory with settings.json.
if [ "$#" -lt 4 ]; then
  printf '%s\n' 'Usage: run_dynamics.sh DATA MODEL OUTPUT EXCLUDE [TRAIN=64] [TEST=32]' >&2
  exit 2
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python -u "$project_root/main.py" dynamics --stage prepare \
  --dataset "$1" --model "$2" --output "$3" --exclude-output "$4" \
  --reference-count "${5:-64}" --limit "${6:-32}" --resume &&
python -u "$project_root/main.py" dynamics --stage run \
  --reference-output "$3/reference" --output "$3/test" --resume
