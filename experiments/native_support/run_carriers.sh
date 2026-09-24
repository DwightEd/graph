#!/usr/bin/env bash
# Full native gradients and finite deletions; old contrast scores are reused.
python main.py transport-carriers \
  --input outputs/native_support_ragtruth4/evidence_contrast_v1 \
  --output outputs/native_support_ragtruth4/message_carriers_v1 \
  --top-k 8 --resume "$@"
