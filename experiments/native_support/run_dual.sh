#!/usr/bin/env bash
# CPU only: preserve current and persistent states from the existing full capture.
python -u main.py transport-dual \
  --input outputs/native_support_ragtruth4/observable_transport_v1 \
  --output outputs/native_support_ragtruth4/dual_heads_v1 \
  --features heads --layers middle --window 16 "$@"
