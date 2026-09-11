#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 AUDIT_ROOT OUTPUT_DIR [BOOTSTRAP]" >&2
  exit 2
fi

audit_root=$1
output_dir=$2
bootstrap=${3:-200}

if [[ ! -f "$audit_root/index.json" ]]; then
  echo "attention-audit index does not exist: $audit_root/index.json" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "output already exists; choose a new directory: $output_dir" >&2
  exit 2
fi

python -u -m control_graph.cli audit-score \
  --audit "$audit_root" \
  --output "$output_dir/scores" \
  --completed-only

python -u -m control_graph.cli audit-evaluate \
  --audit "$audit_root" \
  --scores "$output_dir/scores/scores.jsonl" \
  --output "$output_dir/evaluation.json" \
  --bootstrap "$bootstrap"

echo "evaluation: $output_dir/evaluation.json"
