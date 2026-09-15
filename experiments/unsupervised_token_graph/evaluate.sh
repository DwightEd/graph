#!/usr/bin/env bash
# Evaluate saved scores only. Default: the test output from run_all.sh.
# SPLIT=train selects the train output; explicit OUTPUT remains an exact path.
# ANNOTATIONS is optional when the saved population settings supply its path.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_npz_v2/$SPLIT}"
ANNOTATIONS="${ANNOTATIONS:-}"

args=(--predictions "$OUTPUT" --output "$OUTPUT/evaluation.json"
      --split "$SPLIT" --bootstrap "${BOOTSTRAP:-200}"
      --channel-quantile "${CHANNEL_QUANTILE:-0.9}")
if [[ -n "$ANNOTATIONS" ]]; then args+=(--annotations "$ANNOTATIONS"); fi

"$PY" -u -m experiments.unsupervised_token_graph.run evaluate "${args[@]}" "$@"
