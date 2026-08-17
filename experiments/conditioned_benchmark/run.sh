#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: bash experiments/conditioned_benchmark/run.sh TEST_SPLIT OUTPUT_DIR NAME=SCORES.npz [NAME=SCORES.npz ...]" >&2
  exit 2
fi

TEST_SPLIT=$1
OUTPUT_DIR=$2
shift 2

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPOSITORY_ROOT"

arguments=(
  --split-root "$TEST_SPLIT"
  --output-dir "$OUTPUT_DIR"
  --device "${DEVICE:-cpu}"
  --evaluation-unit "${EVALUATION_UNIT:-token}"
  --ratio-mode "${RATIO_MODE:-reweight}"
  --bootstrap "${BOOTSTRAP:-200}"
  --seed "${SEED:-20260817}"
)

for artifact in "$@"; do
  arguments+=(--artifact "$artifact")
done
for task in ${TASK_TYPES:-all each}; do
  arguments+=(--task-type "$task")
done
for rate in ${POSITIVE_RATES:-native 0.01 0.03 0.05 0.10 0.25 0.50}; do
  arguments+=(--positive-rate "$rate")
done
for metric in ${METRICS:-auroc auprc auprc_lift tpr_at_fpr_05 tpr_at_fpr_10 partial_auroc_fpr_10}; do
  arguments+=(--metric "$metric")
done

python -m experiments.conditioned_benchmark.main "${arguments[@]}"
