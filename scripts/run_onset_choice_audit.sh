#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 AUDIT_ROOT OUTPUT_DIR [PRE_WINDOW] [MATCH_WINDOW] [BOOTSTRAP]" >&2
  exit 2
fi

audit_root=$1
output_dir=$2
pre_window=${3:-3}
match_window=${4:-64}
bootstrap=${5:-200}

if [[ ! -f "$audit_root/index.json" ]]; then
  echo "attention-audit index does not exist: $audit_root/index.json" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "output already exists; choose a new directory: $output_dir" >&2
  exit 2
fi

python -u -m control_graph.cli onset-audit \
  --audit "$audit_root" \
  --output "$output_dir" \
  --pre-window "$pre_window" \
  --match-window "$match_window" \
  --bootstrap "$bootstrap"

echo "onset audit: $output_dir/summary.json"
