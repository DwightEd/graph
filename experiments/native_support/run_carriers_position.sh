#!/usr/bin/env bash
# CPU only. Fixed beta=1, no tuning; the attention matrix is never changed.
python main.py transport-carriers --stage position \
  --input outputs/native_support_ragtruth4/message_carriers_v1 \
  --output outputs/native_support_ragtruth4/message_carriers_position_v1 \
  --position-beta 1 --resume "$@"
