#!/usr/bin/env bash
# CPU-only joint readout of the completed token-carrier experiment.
python main.py transport-unified \
  --input outputs/native_support_ragtruth4/message_carriers_token_v2 \
  --output outputs/native_support_ragtruth4/unified_route_source_v1 \
  --graph-strength 1 --resume "$@"
