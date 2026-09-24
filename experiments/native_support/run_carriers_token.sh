#!/usr/bin/env bash
# Every output token gets its own fixed-foil objective and query-specific cuts.
python main.py transport-carriers --mode token \
  --input outputs/native_support_ragtruth4/message_carriers_v1 \
  --output outputs/native_support_ragtruth4/message_carriers_token_v2 \
  --top-k 8 --receiver-budget 2 --selection conditional --resume "$@"
