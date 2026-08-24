#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_SPLIT:?set TRAIN_SPLIT to the canonical/formal train split}"
: "${TEST_SPLIT:?set TEST_SPLIT to the canonical/formal test split}"

METHOD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${METHOD_ROOT}/../.." && pwd)"
OUT="${OUT:-${METHOD_ROOT}/outputs/full}"
DEVICE="${DEVICE:-cpu}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
REFERENCE_SIZE="${REFERENCE_SIZE:-12000}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-200}"
SEED="${SEED:-42}"

mkdir -p "${OUT}"
cd "${REPOSITORY_ROOT}"

fit_args=(
  --train-split "${TRAIN_SPLIT}"
  --reference "${OUT}/reference.npz"
  --device "${DEVICE}"
  --reference-size "${REFERENCE_SIZE}"
  --seed "${SEED}"
)
if [[ -n "${TRAIN_LIMIT}" ]]; then
  fit_args+=(--limit "${TRAIN_LIMIT}")
fi
python -m experiments.causal_typed_path_debruijn.main fit "${fit_args[@]}"

score_args=(
  --test-split "${TEST_SPLIT}"
  --reference "${OUT}/reference.npz"
  --output "${OUT}/test_scores.npz"
  --device "${DEVICE}"
)
if [[ -n "${TEST_LIMIT}" ]]; then
  score_args+=(--limit "${TEST_LIMIT}")
fi
if [[ "${SAVE_CHANNEL_SIDECARS:-0}" == "1" ]]; then
  score_args+=(--save-channel-sidecars --sidecar-dir "${OUT}/channel_sidecars")
fi
python -m experiments.causal_typed_path_debruijn.main score "${score_args[@]}"

python -m experiments.causal_typed_path_debruijn.main evaluate \
  --test-split "${TEST_SPLIT}" \
  --scores "${OUT}/test_scores.npz" \
  --output "${OUT}/evaluation.json" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES}" \
  --seed "${SEED}"
