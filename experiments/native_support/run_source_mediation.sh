#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-mediation \
  --mode native \
  --input "${MEDIATION_INPUT:-outputs/native_support_ragtruth4/message_carriers_token_v2}" \
  --output "${MEDIATION_OUTPUT:-outputs/native_support_ragtruth4/source_mediation_v1}" \
  --resume "$@"
