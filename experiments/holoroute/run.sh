#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SPLIT:?set TRAIN_SPLIT}"
: "${TEST_SPLIT:?set TEST_SPLIT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-"${ROOT}/experiments/holoroute/outputs/full"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK_TYPE=${TASK_TYPE:-QA}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
EPOCHS=${EPOCHS:-8}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-500}

mkdir -p "${OUT}"
cd "${ROOT}"

train_args=(
  --train-split "${TRAIN_SPLIT}"
  --checkpoint "${OUT}/model.pt"
  --density "${OUT}/density.npz"
  --task-type "${TASK_TYPE}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
)
[[ -z "${TRAIN_LIMIT}" ]] || train_args+=(--limit "${TRAIN_LIMIT}")
"${PYTHON}" -m experiments.holoroute.main train "${train_args[@]}"

score_args=(
  --test-split "${TEST_SPLIT}"
  --checkpoint "${OUT}/model.pt"
  --density "${OUT}/density.npz"
  --output "${OUT}/scores.npz"
  --task-type "${TASK_TYPE}"
  --device "${DEVICE}"
)
[[ -z "${TEST_LIMIT}" ]] || score_args+=(--limit "${TEST_LIMIT}")
"${PYTHON}" -m experiments.holoroute.main score "${score_args[@]}"

"${PYTHON}" -m experiments.holoroute.main evaluate \
  --test-split "${TEST_SPLIT}" \
  --scores "${OUT}/scores.npz" \
  --output-dir "${OUT}/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --device "${DEVICE}"
