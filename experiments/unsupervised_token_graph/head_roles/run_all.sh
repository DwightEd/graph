#!/usr/bin/env bash
cd -- "$(dirname -- "$0")/../../.." || exit
python -u -m experiments.unsupervised_token_graph.head_roles \
  --phase all --output "${OUTPUT:-outputs/head_roles_v1}" --resume "$@"
