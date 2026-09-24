#!/usr/bin/env bash
# Four conditions of the existing observed answers; preserves every saved baseline.
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-contrast \
  --input outputs/native_support_ragtruth4/observable_transport_v1 \
  --output outputs/native_support_ragtruth4/evidence_contrast_v1 \
  --resume "$@"
