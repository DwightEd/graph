#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to a dataset root containing train/ and test/}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${DATA_ROOT}/train}"
TEST_SPLIT="${TEST_SPLIT:-${DATA_ROOT}/test}"
OUT="${OUT:-experiments/causal_walk_audit/outputs/run}"
DEVICE="${DEVICE:-cpu}"
TASK_TYPE="${TASK_TYPE:-QA}"

mkdir -p "${OUT}/evaluation"

COMMON=(
  --device "${DEVICE}"
  --task-type "${TASK_TYPE}"
  --block-rows "${BLOCK_ROWS:-4096}"
  --recent-lag "${RECENT_LAG:-4}"
  --alpha "${ALPHA:-0.5}"
  --backoff-tau "${BACKOFF_TAU:-32}"
  --cusum-slack "${CUSUM_SLACK:-0.5}"
  --rupture-decay "${RUPTURE_DECAY:-0.95}"
  --closure-decay "${CLOSURE_DECAY:-0.9}"
  --channel-fraction "${CHANNEL_FRACTION:-0.2}"
  --fusion-fraction "${FUSION_FRACTION:-0.2}"
  --reservoir-rows "${RESERVOIR_ROWS:-20000}"
  --topology-min-changed-fraction "${TOPOLOGY_MIN_CHANGED_FRACTION:-0.5}"
  --seed "${SEED:-20260825}"
)

FIT_ARGS=("${COMMON[@]}")
SCORE_ARGS=("${COMMON[@]}")
[[ -n "${TRAIN_LIMIT:-}" ]] && FIT_ARGS+=(--limit "${TRAIN_LIMIT}")
[[ -n "${TEST_LIMIT:-}" ]] && SCORE_ARGS+=(--limit "${TEST_LIMIT}")

"${PYTHON}" -m experiments.causal_walk_audit.main fit \
  --train-split "${TRAIN_SPLIT}" \
  --reference "${OUT}/reference.npz" \
  "${FIT_ARGS[@]}"

"${PYTHON}" -m experiments.causal_walk_audit.main score \
  --test-split "${TEST_SPLIT}" \
  --reference "${OUT}/reference.npz" \
  --output "${OUT}/test_scores.npz" \
  "${SCORE_ARGS[@]}"

"${PYTHON}" -m experiments.causal_walk_audit.main evaluate \
  --test-split "${TEST_SPLIT}" \
  --scores "${OUT}/test_scores.npz" \
  --output-dir "${OUT}/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED:-20260825}"
