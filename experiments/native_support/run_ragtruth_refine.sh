#!/usr/bin/env bash
# Reuse completed scalar measurements. No tokenizer, GPU, or model forward.
cd "$(dirname "$0")/../.." || exit 1
python -u main.py transport-refine \
  --input "${REFINE_INPUT:-outputs/native_support_ragtruth_all/source_first_v1}" \
  --output "${REFINE_OUTPUT:-outputs/native_support_ragtruth_all/source_refine_v2}" \
  --select-on-train --resume "$@"
