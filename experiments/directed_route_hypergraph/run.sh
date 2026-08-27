#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
EPOCHS=${EPOCHS:-8}
ROWS_PER_GRAPH=${ROWS_PER_GRAPH:-256}
INCIDENCE_DROPOUT=${INCIDENCE_DROPOUT:-0.15}
HEAD_DROPOUT=${HEAD_DROPOUT:-0.05}
FLOW_WEIGHT=${FLOW_WEIGHT:-0.5}
RESIDUAL_WEIGHT=${RESIDUAL_WEIGHT:-1.0}
VARIANT=${VARIANT:-real}
SEED=${SEED:-20260827}
OUT=${OUT:-${ROOT}/experiments/directed_route_hypergraph/outputs/${TASK,,}/flow_dae_${VARIANT}_seed${SEED}}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
EVALUATE=${EVALUATE:-1}

if [ -z "${TRAIN_SPLIT:-}" ] || [ -z "${TEST_SPLIT:-}" ]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
[ -n "${TRAIN_LIMIT}" ] && TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
[ -n "${TEST_LIMIT}" ] && TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")

run_stage() {
  echo
  echo "$1"
  shift
  "$@" || exit $?
}

run_stage "[1/5] Fit label-free directed hypergraph encoder" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run fit \
  --train "${TRAIN_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --task "${TASK}" --epochs "${EPOCHS}" \
  --rows-per-graph "${ROWS_PER_GRAPH}" --variant "${VARIANT}" \
  --incidence-dropout "${INCIDENCE_DROPOUT}" \
  --head-dropout "${HEAD_DROPOUT}" --flow-weight "${FLOW_WEIGHT}" \
  --residual-weight "${RESIDUAL_WEIGHT}" \
  --seed "${SEED}" --device "${DEVICE}" "${TRAIN_LIMIT_ARGUMENT[@]}"

run_stage "[2/5] Export calibration node embeddings" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run encode \
  --data "${TRAIN_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --output "${OUT}/calibration" --scope calibration \
  --task "${TASK}" --device "${DEVICE}"

run_stage "[3/5] Export test node embeddings" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run encode \
  --data "${TEST_SPLIT}" --checkpoint "${OUT}/model.pt" \
  --output "${OUT}/test" --scope all --task "${TASK}" \
  --device "${DEVICE}" "${TEST_LIMIT_ARGUMENT[@]}"

run_stage "[4/5] Fit node-only PCA-kNN and freeze scores" \
  "${PYTHON}" -m experiments.directed_route_hypergraph.run detect \
  --calibration "${OUT}/calibration/index.npz" \
  --test "${OUT}/test/index.npz" \
  --reference "${OUT}/detector.npz" --scores "${OUT}/scores.npz" \
  --seed "${SEED}"

if [ "${EVALUATE}" = "1" ]; then
  run_stage "[5/5] Evaluate frozen token scores" \
    "${PYTHON}" -m experiments.directed_route_hypergraph.run evaluate \
    --test "${TEST_SPLIT}" --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" --seed "${SEED}"
else
  echo
  echo "[5/5] Labels remain closed"
fi

echo
echo "Finished: ${OUT}"
