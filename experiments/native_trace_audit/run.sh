#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$PWD/teaching/state_audit/src${PYTHONPATH:+:$PYTHONPATH}"
args=()
if [ -n "${MODEL:-}" ]; then
    args+=(--model "$MODEL")
fi
python -m experiments.native_trace_audit.run --resume "${args[@]}" "$@"
