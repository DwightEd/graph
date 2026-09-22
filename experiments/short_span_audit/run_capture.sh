#!/usr/bin/env bash
# Reuse teaching's native model adapter; natural labels select audit targets only.
export PYTHONPATH="$PWD/teaching/state_audit/src${PYTHONPATH:+:$PYTHONPATH}"
python -m experiments.short_span_audit.capture \
  --audit "${AUDIT:-outputs/short_span_audit_v1}" \
  --model "${MODEL:?Set MODEL to the observer checkpoint used by the attention cache}" "$@"
