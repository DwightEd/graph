#!/usr/bin/env bash
# Reuse the completed four-condition capture; no GPU/model work.
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-contrast --stage analyze \
  --input outputs/native_support_ragtruth4/evidence_contrast_v1 \
  --output outputs/native_support_ragtruth4/evidence_contrast_aggregation_v1 \
  --resume "$@"
