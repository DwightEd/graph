#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
MODEL_PATH=${MODEL_PATH:-}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
SEED=${SEED:-20260828}
IMPUTATION=${IMPUTATION:-zero}
LOAD_DTYPE=${LOAD_DTYPE:-bfloat16}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-float32}
BLOCK_HEADS=${BLOCK_HEADS:-4}
SAVE_BASIS=${SAVE_BASIS:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
LIMIT=${LIMIT:-}
BOOTSTRAP=${BOOTSTRAP:-500}
CV_FOLDS=${CV_FOLDS:-5}
START_STAGE=${START_STAGE:-1}
FORCE_OPERATORS=${FORCE_OPERATORS:-0}

if [ -z "${MODEL_PATH}" ]; then
  echo "MODEL_PATH must point to the same frozen LLM used to produce the attention cache."
  exit 1
fi
if ! [[ "${START_STAGE}" =~ ^[1-3]$ ]]; then
  echo "START_STAGE must be 1, 2, or 3."
  exit 1
fi

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_operator_validation/outputs/${TASK,,}/${MODEL_TAG}_${IMPUTATION}}
OPERATOR_CACHE=${OPERATOR_CACHE:-${REPO}/experiments/attention_operator_validation/outputs/operators/${MODEL_TAG}/operator_geometry.pt}
BASIS_DIR=${BASIS_DIR:-${REPO}/experiments/attention_operator_validation/outputs/operators/${MODEL_TAG}/basis}

cd "${REPO}" || exit 1
mkdir -p "${OUT}" "$(dirname "${OPERATOR_CACHE}")"

LIMIT_ARGUMENT=()
[ -n "${LIMIT}" ] && LIMIT_ARGUMENT=(--limit "${LIMIT}")
TRUST_ARGUMENT=()
[ "${TRUST_REMOTE_CODE}" = "1" ] && TRUST_ARGUMENT=(--trust-remote-code)
BASIS_ARGUMENT=()
[ "${SAVE_BASIS}" = "1" ] && BASIS_ARGUMENT=(--basis-dir "${BASIS_DIR}")

if [ "${START_STAGE}" -le 1 ]; then
  if [ -f "${OPERATOR_CACHE}" ] && [ "${FORCE_OPERATORS}" != "1" ]; then
    echo
    echo "[1/3] Reuse cached operator geometry: ${OPERATOR_CACHE}"
  else
    echo
    echo "[1/3] Extract reusable W_O W_V operator geometry"
    "${PYTHON}" -m experiments.attention_operator_validation.run operators \
      --model "${MODEL_PATH}" \
      --output "${OPERATOR_CACHE}" \
      --device "${DEVICE}" \
      --load-dtype "${LOAD_DTYPE}" \
      --compute-dtype "${COMPUTE_DTYPE}" \
      --block-heads "${BLOCK_HEADS}" \
      "${BASIS_ARGUMENT[@]}" \
      "${TRUST_ARGUMENT[@]}"
  fi
else
  echo
  echo "[1/3] Operator extraction -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${OPERATOR_CACHE}" ]; then
  echo "Operator cache is missing: ${OPERATOR_CACHE}"
  exit 1
fi

if [ "${START_STAGE}" -le 2 ]; then
  echo
  echo "[2/3] Freeze label-free answer-level pair-code features"
  "${PYTHON}" -m experiments.attention_operator_validation.run features \
    --data "${TEST_SPLIT}" \
    --operators "${OPERATOR_CACHE}" \
    --output "${OUT}/features.npz" \
    --task "${TASK}" \
    --imputation "${IMPUTATION}" \
    --seed "${SEED}" \
    "${LIMIT_ARGUMENT[@]}"
else
  echo
  echo "[2/3] Feature extraction -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${OUT}/features.npz" ]; then
  echo "Feature artifact is missing: ${OUT}/features.npz"
  exit 1
fi

echo
echo "[3/3] Open token labels and evaluate answer-level mechanisms"
"${PYTHON}" -m experiments.attention_operator_validation.run evaluate \
  --data "${TEST_SPLIT}" \
  --features "${OUT}/features.npz" \
  --output "${OUT}/evaluation.json" \
  --bootstrap "${BOOTSTRAP}" \
  --cv-folds "${CV_FOLDS}" \
  --seed "${SEED}"

echo
echo "Finished: ${OUT}"
echo "Operator cache: ${OPERATOR_CACHE}"
