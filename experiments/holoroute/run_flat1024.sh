#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SPLIT:?set TRAIN_SPLIT}"
: "${TEST_SPLIT:?set TEST_SPLIT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-"${ROOT}/experiments/holoroute/outputs/flat_1024"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK_TYPE=${TASK_TYPE:-QA}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
EPOCHS=${EPOCHS:-8}
HIDDEN_DIM=${HIDDEN_DIM:-96}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-500}

mkdir -p "${OUT}"
cd "${ROOT}"

train_args=(
  --train-split "${TRAIN_SPLIT}"
  --checkpoint "${OUT}/flat1024_model.pt"
  --density "${OUT}/flat1024_density.npz"
  --task-type "${TASK_TYPE}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --hidden-dim "${HIDDEN_DIM}"
)
[[ -z "${TRAIN_LIMIT}" ]] || train_args+=(--limit "${TRAIN_LIMIT}")
"${PYTHON}" -m experiments.holoroute.flat1024_main train "${train_args[@]}"

score_args=(
  --test-split "${TEST_SPLIT}"
  --checkpoint "${OUT}/flat1024_model.pt"
  --density "${OUT}/flat1024_density.npz"
  --output "${OUT}/flat1024_scores.npz"
  --task-type "${TASK_TYPE}"
  --device "${DEVICE}"
)
[[ -z "${TEST_LIMIT}" ]] || score_args+=(--limit "${TEST_LIMIT}")
"${PYTHON}" -m experiments.holoroute.flat1024_main score "${score_args[@]}"

"${PYTHON}" -m experiments.holoroute.flat1024_main evaluate \
  --test-split "${TEST_SPLIT}" \
  --scores "${OUT}/flat1024_scores.npz" \
  --output-dir "${OUT}/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --device "${DEVICE}"
