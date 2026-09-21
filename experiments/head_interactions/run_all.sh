#!/usr/bin/env bash
cd -- "$(dirname -- "$0")/../.." || exit
export PYTHONPATH="$PWD/teaching/state_audit/src${PYTHONPATH:+:$PYTHONPATH}"

python -u -m experiments.head_interactions.run \
  --output "${OUTPUT:-outputs/head_interactions_v2}" --resume "$@"
