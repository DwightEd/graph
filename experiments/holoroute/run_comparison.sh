#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SPLIT:?set TRAIN_SPLIT}"
: "${TEST_SPLIT:?set TEST_SPLIT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT=${OUT_ROOT:-"${ROOT}/experiments/holoroute/outputs/comparison"}

OUT="${OUT_ROOT}/full" bash "${ROOT}/experiments/holoroute/run.sh"
OUT="${OUT_ROOT}/flat_1024" bash "${ROOT}/experiments/holoroute/run_flat1024.sh"

echo "HoloRoute:  ${OUT_ROOT}/full/evaluation/evaluation.json"
echo "flat-1024: ${OUT_ROOT}/flat_1024/evaluation/evaluation.json"
