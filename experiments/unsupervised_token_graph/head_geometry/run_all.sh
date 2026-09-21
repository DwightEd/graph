#!/usr/bin/env bash
cd -- "$(dirname -- "$0")/../../.." || exit
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase all --tasks QA --output "${OUTPUT:-outputs/head_geometry_v2}" --resume "$@"
