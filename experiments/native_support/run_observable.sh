#!/usr/bin/env bash
# All tokens of the existing small input manifest; no dataset expansion.
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-observable \
  --input outputs/native_support_ragtruth4 \
  --output outputs/native_support_ragtruth4/observable_transport_v1 \
  --resume "$@"
