#!/usr/bin/env bash
# Existing full observable cache -> supervised source-held-out readout; CPU only.
python -u main.py transport-readout \
  --input outputs/native_support_ragtruth4/observable_transport_v1 \
  --output outputs/native_support_ragtruth4/readout_heads_v1 \
  --features heads --layers middle --models logistic "$@"
