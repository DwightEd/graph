#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
CACHE_ROOT=${CACHE_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
SOURCE_INFO=${SOURCE_INFO:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
TOKEN_CHUNK=${TOKEN_CHUNK:-128}
INTERVENTION_BATCH=${INTERVENTION_BATCH:-3}
TOP_K=${TOP_K:-8}
LOGIT_CHUNK=${LOGIT_CHUNK:-64}
TRACE_LEVEL=${TRACE_LEVEL:-mechanism}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-20260828}
LIMIT=${LIMIT:-}

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/qa/${MODEL_TAG}_teacher_forced_seed${SEED}}
TRAIN_ROOT=${TRAIN_ROOT:-${CACHE_ROOT}/train}
TEST_ROOT=${TEST_ROOT:-${CACHE_ROOT}/test}
TRAIN_TRACES=${TRAIN_TRACES:-${OUT}/train/traces}
TEST_TRACES=${TEST_TRACES:-${OUT}/test/traces}

cd "${REPO}" || exit $?
mkdir -p "${OUT}" || exit $?

if [ -f "${OUT}/traces/index.jsonl" ] && [ ! -f "${TEST_TRACES}/index.jsonl" ]; then
  TEST_TRACES=${OUT}/traces
fi

LIMIT_ARGUMENT=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGUMENT=(--limit "${LIMIT}")
fi

capture_shard() {
  NAME=$1
  CACHE=$2
  TRACES=$3
  echo "cache shard: ${NAME}"
  "${PYTHON}" -m experiments.attention_mechanism_audit.run capture \
    --split-root "${CACHE}" \
    --source-info "${SOURCE_INFO}" \
    --model "${MODEL_PATH}" \
    --output "${TRACES}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --predictor-chunk "${TOKEN_CHUNK}" \
    --intervention-batch "${INTERVENTION_BATCH}" \
    --top-k "${TOP_K}" \
    --logit-chunk "${LOGIT_CHUNK}" \
    --trace-level "${TRACE_LEVEL}" \
    "${LIMIT_ARGUMENT[@]}"
  STATUS=$?
  if [ "${STATUS}" -ne 0 ]; then
    echo "ERROR: ${NAME} capture failed with exit code ${STATUS}." >&2
    exit "${STATUS}"
  fi
}

echo "[1/2] Capture/resume every cached QA response"
capture_shard train "${TRAIN_ROOT}" "${TRAIN_TRACES}"
capture_shard test "${TEST_ROOT}" "${TEST_TRACES}"

echo
echo "[2/2] Pool every cached QA token and evaluate once"
"${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
  --input "${TRAIN_TRACES}" "${TRAIN_ROOT}" \
  --input "${TEST_TRACES}" "${TEST_ROOT}" \
  --output "${OUT}/report.json" \
  --bootstrap "${BOOTSTRAP}" \
  --seed "${SEED}"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: all-data evaluation failed with exit code ${STATUS}." >&2
  exit "${STATUS}"
fi

echo
echo "report: ${OUT}/report.json"
echo "token scores: ${OUT}/token_scores.npz"
echo "population figures: ${OUT}/figures"
