#!/usr/bin/env bash
# CPU-only token readout; reuses completed native measurements without new forwards.
python main.py transport-unified \
  --input outputs/native_support_ragtruth4/message_carriers_token_v2 \
  --output outputs/native_support_ragtruth4/unified_token_v2 \
  --token-readout --anchor-strength 1 --graph-strength 1 --continuity-strength 0.05 \
  --resume "$@"
