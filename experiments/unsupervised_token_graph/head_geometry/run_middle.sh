#!/usr/bin/env bash
cd -- "$(dirname -- "$0")/../../.." || exit

# Llama-3.1-8B: use only the middle half of layers; no TEST-based head ranking.
# Explicit --layers / --heads / --tasks after this preset override these choices.
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase all --tasks QA Summary Data2txt --layers {8..23} --no-layer-bands \
  --output "${OUTPUT:-outputs/head_geometry_middle}" --resume "$@"
