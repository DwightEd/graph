#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SPLIT:?set TRAIN_SPLIT}"
: "${TEST_SPLIT:?set TEST_SPLIT}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-"${ROOT}/experiments/holoroute/outputs/run"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
MODEL=${MODEL:-holoroute}

mkdir -p "${OUT}"
cd "${ROOT}"

train_command=train
score_command=score
if [[ "${MODEL}" == "flat1024" ]]; then
  train_command=flat-train
  score_command=flat-score
fi

train=(
  "${PYTHON}" -m experiments.holoroute.run "${train_command}"
  --train "${TRAIN_SPLIT}"
  --checkpoint "${OUT}/model.pt"
  --reference "${OUT}/reference.npz"
  --task "${TASK}"
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
)
[[ -z "${TRAIN_LIMIT}" ]] || train+=(--limit "${TRAIN_LIMIT}")
"${train[@]}"

score=(
  "${PYTHON}" -m experiments.holoroute.run "${score_command}"
  --test "${TEST_SPLIT}"
  --checkpoint "${OUT}/model.pt"
  --reference "${OUT}/reference.npz"
  --output "${OUT}/scores.npz"
  --task "${TASK}"
  --device "${DEVICE}"
)
[[ -z "${TEST_LIMIT}" ]] || score+=(--limit "${TEST_LIMIT}")
"${score[@]}"

"${PYTHON}" -m experiments.holoroute.run evaluate \
  --test "${TEST_SPLIT}" \
  --scores "${OUT}/scores.npz" \
  --output "${OUT}/evaluation" \
  --device "${DEVICE}"
