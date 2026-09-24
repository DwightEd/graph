#!/usr/bin/env bash
# One process: capture every selected original token, report coverage, auto-pack.
# Run from any directory. Extra CLI flags override the defaults below.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." || exit 1

exec "${FUNCTION_AUDIT_PYTHON:-python}" -u main.py transport-functions \
  --stage run \
  --input outputs/native_support_ragtruth4 \
  --output outputs/native_support_ragtruth4/function_audit_v3 \
  --response-ids 12219 \
  --device cuda:0 \
  --dtype bfloat16 \
  --resume \
  "$@"
