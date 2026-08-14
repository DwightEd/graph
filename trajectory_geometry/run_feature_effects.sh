#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 TRAIN_FEATURE_DIR TEST_FEATURE_DIR [OUTPUT_DIR]" >&2
  exit 2
fi

TRAIN_FEATURE_DIR=$1
TEST_FEATURE_DIR=$2
OUTPUT_DIR=${3:-"$(dirname "$TEST_FEATURE_DIR")/gate_a_effects"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

cd "$SCRIPT_DIR"
echo "train_features=$TRAIN_FEATURE_DIR"
echo "test_features=$TEST_FEATURE_DIR"
echo "output_dir=$OUTPUT_DIR"
python -m trajectory_geometry.cli evaluate \
  --train-features "$TRAIN_FEATURE_DIR" \
  --test-features "$TEST_FEATURE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "${DEVICE:-cuda}"
