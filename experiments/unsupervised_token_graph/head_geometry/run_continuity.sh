#!/usr/bin/env bash
# Read frozen predictions only; existing model outputs and thresholds are untouched.
python -m experiments.unsupervised_token_graph.head_geometry.continuity \
  --output "${OUTPUT:-outputs/head_cross_terms_v1}" "$@"
