#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
CACHE_ROOT=${CACHE_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
TRAIN_ROOT=${TRAIN_ROOT:-${CACHE_ROOT}/train}
TEST_ROOT=${TEST_ROOT:-${CACHE_ROOT}/test}
SOURCE_INFO=${SOURCE_INFO:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl}
SPLITS=${SPLITS:-"train test"}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
DTYPE=${DTYPE:-bfloat16}
TOKEN_CHUNK=${TOKEN_CHUNK:-128}
INTERVENTION_BATCH=${INTERVENTION_BATCH:-3}
TOP_K=${TOP_K:-8}
LOGIT_CHUNK=${LOGIT_CHUNK:-64}
TRACE_LEVEL=${TRACE_LEVEL:-mechanism}
POSITION_BIN=${POSITION_BIN:-16}
BOOTSTRAP=${BOOTSTRAP:-10000}
SEED=${SEED:-20260828}
LIMIT=${LIMIT:-}

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/qa/${MODEL_TAG}_teacher_forced_seed${SEED}}

cd "${REPO}" || exit $?
mkdir -p "${OUT}" || exit $?

LIMIT_ARGUMENT=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGUMENT=(--limit "${LIMIT}")
fi

COMBINE_ARGUMENTS=()
for SPLIT in ${SPLITS}; do
  if [ "${SPLIT}" = "train" ]; then
    SPLIT_ROOT=${TRAIN_ROOT}
  elif [ "${SPLIT}" = "test" ]; then
    SPLIT_ROOT=${TEST_ROOT}
  else
    echo "ERROR: unsupported split ${SPLIT}; use train and/or test." >&2
    exit 2
  fi

  SPLIT_OUT=${OUT}/${SPLIT}
  TRACES=${SPLIT_OUT}/traces
  if [ "${SPLIT}" = "test" ] && [ -f "${OUT}/traces/index.jsonl" ]; then
    TRACES=${OUT}/traces
  fi
  REPORT=${SPLIT_OUT}/report.json
  mkdir -p "${SPLIT_OUT}" || exit $?

  echo "[${SPLIT} 1/2] Capture/resume every cached QA response"
  "${PYTHON}" -m experiments.attention_mechanism_audit.run capture \
    --split-root "${SPLIT_ROOT}" \
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
    echo "ERROR: ${SPLIT} capture failed with exit code ${STATUS}." >&2
    exit "${STATUS}"
  fi

  echo
  echo "[${SPLIT} 2/2] Build every sample audit and split-level figures"
  "${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
    --traces "${TRACES}" \
    --split-root "${SPLIT_ROOT}" \
    --split-name "${SPLIT}" \
    --model "${MODEL_PATH}" \
    --output "${REPORT}" \
    --sample-output "${SPLIT_OUT}/sample_audits" \
    --figures "${SPLIT_OUT}/figures" \
    --position-bin "${POSITION_BIN}" \
    --bootstrap "${BOOTSTRAP}" \
    --seed "${SEED}"
  STATUS=$?
  if [ "${STATUS}" -ne 0 ]; then
    echo "ERROR: ${SPLIT} evaluation failed with exit code ${STATUS}." >&2
    exit "${STATUS}"
  fi
  COMBINE_ARGUMENTS+=(--input "${SPLIT}" "${REPORT}")
done

echo
echo "[all] Recompute train/test/combined source-level results and figures"
mkdir -p "${OUT}/all" || exit $?
"${PYTHON}" -m experiments.attention_mechanism_audit.run combine \
  "${COMBINE_ARGUMENTS[@]}" \
  --output "${OUT}/all/report.json" \
  --figures "${OUT}/all/figures" \
  --position-bin "${POSITION_BIN}" \
  --bootstrap "${BOOTSTRAP}" \
  --seed "${SEED}"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: combined audit failed with exit code ${STATUS}." >&2
  exit "${STATUS}"
fi

echo
echo "all-QA report: ${OUT}/all/report.json"
echo "per-sample audits: ${OUT}/{train,test}/sample_audits"
echo "figures: ${OUT}/{train,test,all}/figures"
