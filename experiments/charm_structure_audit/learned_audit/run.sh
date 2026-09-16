#!/usr/bin/env bash
# Foreground audit of the EXISTING QA CHARM; never train a replacement detector.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)" || exit
cd "$ROOT" || exit
PY="${PY:-python}"
MODEL_DIR="${MODEL_DIR:-outputs/charm_structure_audit_qa/QA/seed_0/charm_out}"
OUTPUT="${OUTPUT:-$MODEL_DIR/test/learned_audit}"
"$PY" -u -m experiments.charm_structure_audit.diagnose_saved \
  --predictions "$MODEL_DIR/test" --output "$OUTPUT/saved_scores" || exit
"$PY" -u -m experiments.charm_structure_audit.learned_audit.run \
  --model-dir "$MODEL_DIR" \
  --prepared "${PREPARED:-outputs/charm_structure_audit_qa/data}" \
  --output "$OUTPUT" \
  --stage "${STAGE:-all}" \
  --bootstrap "${BOOTSTRAP:-200}" "$@"
