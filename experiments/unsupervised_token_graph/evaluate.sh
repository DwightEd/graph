#!/usr/bin/env bash
# Evaluate saved scores only. ANNOTATIONS is optional when population/settings.json
# already supplied the dataset path during analysis. Never recompute graph scores.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
OUTPUT="${OUTPUT:-outputs/source_carrier_information_npz_v2}"
ANNOTATIONS="${ANNOTATIONS:-}"

args=(--predictions "$OUTPUT" --output "$OUTPUT/evaluation.json"
      --split "${SPLIT:-test}" --bootstrap "${BOOTSTRAP:-200}"
      --channel-quantile "${CHANNEL_QUANTILE:-0.9}")
if [[ -n "$ANNOTATIONS" ]]; then args+=(--annotations "$ANNOTATIONS"); fi

"$PY" -u -m experiments.unsupervised_token_graph.run evaluate "${args[@]}" "$@"
