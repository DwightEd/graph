#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

git pull --ff-only origin main
git log -3 --oneline

conda run --no-capture-output -n research \
  python -m pytest -q \
  experiments/reanchor_flow/message_dag/tests/test_causal_jvp.py \
  experiments/reanchor_flow/message_dag/tests/test_event_execution.py::test_cuda_memory_profiling_passes_integer_device_to_legacy_torch

conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag lookback \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --output experiments/reanchor_flow/outputs/attention_audit_v3/lookback_events_v2 \
  --phase all \
  --split all \
  --task all \
  --completed-only \
  --device cuda:0 \
  --window 10 \
  --gain 0.10 \
  --local-floor 0.50 \
  --query-chunk 8 \
  --event-batch 2 \
  --state-cache-gib 1 \
  --cpu-threads 4 \
  --bootstrap 200 \
  --profile
