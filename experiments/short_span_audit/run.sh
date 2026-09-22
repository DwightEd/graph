#!/usr/bin/env bash
# CPU evaluation of frozen predictions; never loads an LLM or refits its detector.
python -m experiments.short_span_audit \
  --input "${INPUT:-outputs/head_cross_terms_v1}" \
  --output "${OUTPUT:-outputs/short_span_audit_v1}" "$@"
