#!/usr/bin/env bash
# Reuse the complete prepared run; no model or attention-cache rescan is needed.
python -m experiments.unsupervised_token_graph.head_geometry \
  --phase all --suite cross_terms --no-layer-bands \
  --observations-from "${OBSERVATIONS:-outputs/head_geometry_middle}" \
  --output "${OUTPUT:-outputs/head_cross_terms_v1}" --resume "$@"
