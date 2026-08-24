#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to the dataset root containing train/ and test/}"

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cpu}
TASK_TYPE=${TASK_TYPE:-QA}
OUT=${OUT:-experiments/attention_holonomy_audit/outputs/smoke}
TRAIN_LIMIT=${TRAIN_LIMIT:-100}
TEST_LIMIT=${TEST_LIMIT:-30}
SEED=${SEED:-20260825}

mkdir -p "${OUT}"

COMMON=(
  --device "${DEVICE}"
  --task-type "${TASK_TYPE}"
  --seed "${SEED}"
  --block-rows "${BLOCK_ROWS:-4096}"
  --max-relay-predecessors "${MAX_RELAY_PREDECESSORS:-12}"
  --max-query-events "${MAX_QUERY_EVENTS:-32}"
  --reservoir-rows "${RESERVOIR_ROWS:-50000}"
)

"${PYTHON}" -u -m experiments.attention_holonomy_audit.main fit \
  --train-split "${DATA_ROOT}/train" \
  --reference "${OUT}/reference.npz" \
  --limit "${TRAIN_LIMIT}" \
  "${COMMON[@]}" \
  2>&1 | tee "${OUT}/fit.log"

"${PYTHON}" -u -m experiments.attention_holonomy_audit.main score \
  --test-split "${DATA_ROOT}/test" \
  --reference "${OUT}/reference.npz" \
  --output "${OUT}/scores.npz" \
  --sidecar-dir "${OUT}/maps" \
  --limit "${TEST_LIMIT}" \
  "${COMMON[@]}" \
  2>&1 | tee "${OUT}/score.log"

"${PYTHON}" -u -m experiments.attention_holonomy_audit.main evaluate \
  --test-split "${DATA_ROOT}/test" \
  --scores "${OUT}/scores.npz" \
  --output-dir "${OUT}/evaluation" \
  --device "${DEVICE}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT}/evaluate.log"
