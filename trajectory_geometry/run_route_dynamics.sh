#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 ATTENTION_CACHE_ROOT OUTPUT_DIR [train|test]" >&2
  exit 2
fi

ATTENTION_CACHE_ROOT=$1
OUTPUT_DIR=$2
SPLIT=${3:-train}

ARGS=(
  --attention-root "$ATTENTION_CACHE_ROOT"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --embedding-dim "${EMBEDDING_DIM:-256}"
)
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

python -m trajectory_geometry.cli extract "${ARGS[@]}"
