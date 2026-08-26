#!/usr/bin/env bash

REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
TEST_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
OUT=${OUT:-${REPO}/experiments/information_flow/outputs/qa}
DEVICE=${DEVICE:-cuda}
SKETCH_DIM=${SKETCH_DIM:-32}
RESIDUAL_WEIGHT=${RESIDUAL_WEIGHT:-1.0}
UNRESOLVED=${UNRESOLVED:-self}
CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.20}
TRAIN_LIMIT=${TRAIN_LIMIT:-}
TEST_LIMIT=${TEST_LIMIT:-}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-20}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-20260827}

cd "${REPO}" || exit 1
mkdir -p "${OUT}"

TRAIN_LIMIT_ARGUMENT=()
TEST_LIMIT_ARGUMENT=()
[ -n "${TRAIN_LIMIT}" ] && TRAIN_LIMIT_ARGUMENT=(--limit "${TRAIN_LIMIT}")
[ -n "${TEST_LIMIT}" ] && TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")

echo "[1/3] Extract calibration information-flow embeddings"
python -m experiments.information_flow.run extract \
  --data "${TRAIN_ROOT}" \
  --output "${OUT}/calibration" \
  --scope calibration \
  --task QA \
  --sketch-dim "${SKETCH_DIM}" \
  --residual-weight "${RESIDUAL_WEIGHT}" \
  --unresolved "${UNRESOLVED}" \
  --calibration-fraction "${CALIBRATION_FRACTION}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${TRAIN_LIMIT_ARGUMENT[@]}" || exit $?

echo "[2/3] Extract test information-flow embeddings"
python -m experiments.information_flow.run extract \
  --data "${TEST_ROOT}" \
  --output "${OUT}/test" \
  --scope all \
  --task QA \
  --sketch-dim "${SKETCH_DIM}" \
  --residual-weight "${RESIDUAL_WEIGHT}" \
  --unresolved "${UNRESOLVED}" \
  --calibration-fraction "${CALIBRATION_FRACTION}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${TEST_LIMIT_ARGUMENT[@]}" || exit $?

echo "[3/3] Evaluate with the shared node-only readers"
python -m experiments.information_flow.run evaluate \
  --calibration "${OUT}/calibration" \
  --test "${OUT}/test" \
  --test-root "${TEST_ROOT}" \
  --output "${OUT}/evaluation" \
  --device "${DEVICE}" \
  --folds "${FOLDS}" \
  --epochs "${EPOCHS}" \
  --bootstrap "${BOOTSTRAP}" \
  --seeds "${SEED}" || exit $?

echo "Finished: ${OUT}/evaluation/report.json"
