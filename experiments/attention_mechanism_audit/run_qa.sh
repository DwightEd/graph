#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
RAGTRUTH_ROOT=${RAGTRUTH_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth}
SOURCE_INFO=${SOURCE_INFO:-${RAGTRUTH_ROOT}/source_info.jsonl}
if [ ! -f "${SOURCE_INFO}" ] && [ -f "${RAGTRUTH_ROOT}/dataset/source_info.jsonl" ]; then
  SOURCE_INFO=${RAGTRUTH_ROOT}/dataset/source_info.jsonl
fi
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-auto}
TASK=${TASK:-QA}
SEED=${SEED:-20260828}
VOCAB_CHUNK_SIZE=${VOCAB_CHUNK_SIZE:-4096}
GRADIENT_PROBES=${GRADIENT_PROBES:-8}
ROLE_NULL_BIN_WIDTH=${ROLE_NULL_BIN_WIDTH:-32}
LIMIT=${LIMIT:-}
BOOTSTRAP=${BOOTSTRAP:-1000}
FOLDS=${FOLDS:-5}
START_STAGE=${START_STAGE:-1}
FORCE_ROLES=${FORCE_ROLES:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}

if ! [[ "${START_STAGE}" =~ ^[1-3]$ ]]; then
  echo "START_STAGE must be 1, 2, or 3."
  exit 1
fi

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/${TASK,,}/${MODEL_TAG}_seed${SEED}}
ROLE_INDEX=${ROLE_INDEX:-${OUT}/prompt_roles.jsonl}
ARTIFACT=${ARTIFACT:-${OUT}/mechanisms.npz}
EVALUATION=${EVALUATION:-${OUT}/evaluation.json}

cd "${REPO}" || exit 1
mkdir -p "${OUT}"

LIMIT_ARGUMENT=()
[ -n "${LIMIT}" ] && LIMIT_ARGUMENT=(--limit "${LIMIT}")
TRUST_ARGUMENT=()
[ "${TRUST_REMOTE_CODE}" = "1" ] && TRUST_ARGUMENT=(--trust-remote-code)

if [ "${START_STAGE}" -le 1 ]; then
  if [ ! -f "${SOURCE_INFO}" ]; then
    echo "SOURCE_INFO must point to RAGTruth's label-free source_info.jsonl: ${SOURCE_INFO}"
    exit 1
  fi
  if [ -f "${ROLE_INDEX}" ] && [ "${FORCE_ROLES}" != "1" ]; then
    echo
    echo "[1/3] Reuse exact prompt-role index: ${ROLE_INDEX}"
  else
    echo
    echo "[1/3] Reconstruct and verify label-free prompt roles"
    "${PYTHON}" -m experiments.attention_mechanism_audit.run roles \
      --data "${TEST_SPLIT}" \
      --source-info "${SOURCE_INFO}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --output "${ROLE_INDEX}" \
      --task "${TASK}" \
      "${LIMIT_ARGUMENT[@]}" \
      "${TRUST_ARGUMENT[@]}"
  fi
else
  echo
  echo "[1/3] Prompt-role reconstruction -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${ROLE_INDEX}" ]; then
  echo "Prompt-role index is missing: ${ROLE_INDEX}"
  exit 1
fi

if [ "${START_STAGE}" -le 2 ]; then
  if [ ! -f "${SOURCE_INFO}" ]; then
    echo "SOURCE_INFO is required to bind the frozen mechanism artifact."
    exit 1
  fi
  echo
  echo "[2/3] Replay the frozen model and capture three separate mechanisms"
  "${PYTHON}" -m experiments.attention_mechanism_audit.run capture \
    --data "${TEST_SPLIT}" \
    --roles "${ROLE_INDEX}" \
    --source-info "${SOURCE_INFO}" \
    --model "${MODEL_PATH}" \
    --output "${ARTIFACT}" \
    --device "${DEVICE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --task "${TASK}" \
    --vocab-chunk-size "${VOCAB_CHUNK_SIZE}" \
    --gradient-probes "${GRADIENT_PROBES}" \
    --attribution-seed "${SEED}" \
    --role-null-bin-width "${ROLE_NULL_BIN_WIDTH}" \
    "${LIMIT_ARGUMENT[@]}" \
    "${TRUST_ARGUMENT[@]}"
else
  echo
  echo "[2/3] Mechanism capture -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${ARTIFACT}" ]; then
  echo "Mechanism artifact is missing: ${ARTIFACT}"
  exit 1
fi

echo
echo "[3/3] Freeze artifact bytes, then open labels for post-hoc evaluation"
"${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
  --data "${TEST_SPLIT}" \
  --artifact "${ARTIFACT}" \
  --output "${EVALUATION}" \
  --bootstrap "${BOOTSTRAP}" \
  --folds "${FOLDS}" \
  --seed "${SEED}"

echo
echo "Finished: ${OUT}"
echo "Mechanism artifact: ${ARTIFACT}"
echo "Evaluation: ${EVALUATION}"
