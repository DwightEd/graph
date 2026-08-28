#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
TEST_ROOT=${TEST_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/test}
SOURCE_INFO=${SOURCE_INFO:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
TOKEN_CHUNK=${TOKEN_CHUNK:-128}
INTERVENTION_BATCH=${INTERVENTION_BATCH:-3}
TOP_K=${TOP_K:-8}
LOGIT_CHUNK=${LOGIT_CHUNK:-64}
POSITION_BIN=${POSITION_BIN:-16}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-20260828}
LIMIT=${LIMIT:-}

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/qa/${MODEL_TAG}_teacher_forced_seed${SEED}}
TRACES=${TRACES:-${OUT}/traces}
REPORT=${REPORT:-${OUT}/report.json}

cd "${REPO}" || exit $?
mkdir -p "${OUT}" || exit $?

LIMIT_ARGUMENT=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGUMENT=(--limit "${LIMIT}")
fi

echo "[1/2] Capture chunked A/V functional-message traces"
"${PYTHON}" -m experiments.attention_mechanism_audit.run capture \
  --split-root "${TEST_ROOT}" \
  --source-info "${SOURCE_INFO}" \
  --model "${MODEL_PATH}" \
  --output "${TRACES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --predictor-chunk "${TOKEN_CHUNK}" \
  --intervention-batch "${INTERVENTION_BATCH}" \
  --top-k "${TOP_K}" \
  --logit-chunk "${LOGIT_CHUNK}" \
  "${LIMIT_ARGUMENT[@]}"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: trace capture failed with exit code ${STATUS}." >&2
  exit "${STATUS}"
fi

echo
echo "[2/2] Read labels only now and run position-matched mechanism tests"
"${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
  --traces "${TRACES}" \
  --split-root "${TEST_ROOT}" \
  --output "${REPORT}" \
  --position-bin "${POSITION_BIN}" \
  --bootstrap "${BOOTSTRAP}" \
  --seed "${SEED}"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: post-hoc evaluation failed with exit code ${STATUS}." >&2
  exit "${STATUS}"
fi

echo
echo "traces: ${TRACES}"
echo "report: ${REPORT}"
