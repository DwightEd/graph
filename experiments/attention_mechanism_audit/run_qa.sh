#!/usr/bin/env bash

fail_run() {
  local message=$1
  local code=${2:-1}
  echo >&2
  echo "ERROR: ${message}" >&2
  echo "Stopped immediately; no later stage was run." >&2
  exit "${code}"
}

run_stage() {
  local stage_name=$1
  local stage_status
  shift

  "$@"
  stage_status=$?
  if [ "${stage_status}" -ne 0 ]; then
    fail_run "${stage_name} failed with exit code ${stage_status}." "${stage_status}"
  fi
}

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
PAIRS=${PAIRS:-${REPO}/experiments/attention_mechanism_audit/inputs/qa/audit_pairs.jsonl}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-20260828}
LIMIT=${LIMIT:-}

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/qa/${MODEL_TAG}_grounding_control_seed${SEED}}
ARTIFACT=${ARTIFACT:-${OUT}/control_chain.npz}
REPORT=${REPORT:-${OUT}/report.json}

run_stage "enter repository ${REPO}" cd "${REPO}"

if [ ! -d "${MODEL_PATH}" ]; then
  fail_run "model directory does not exist: ${MODEL_PATH}"
fi
if [ ! -f "${PAIRS}" ]; then
  fail_run "controlled pair manifest does not exist: ${PAIRS}"
fi
run_stage "create output directory ${OUT}" mkdir -p "${OUT}"

LIMIT_ARGUMENT=()
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGUMENT=(--limit "${LIMIT}")
fi

echo "EXPERIMENT: SELECT--RELAY--OVERRIDE grounding control"
echo "MODEL: ${MODEL_PATH}"
echo "PAIRS: ${PAIRS}"
echo "OUTPUT: ${OUT}"

echo
echo "[1/2] Run seven exact frozen-model causal branches"
run_stage "causal audit" \
  "${PYTHON}" -m experiments.attention_mechanism_audit.run audit \
  --pairs "${PAIRS}" \
  --model "${MODEL_PATH}" \
  --output "${ARTIFACT}" \
  --device "${DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  "${LIMIT_ARGUMENT[@]}"

echo
echo "[2/2] Source-grouped label-free mechanism summary"
run_stage "mechanism summary" \
  "${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
  --artifact "${ARTIFACT}" \
  --output "${REPORT}" \
  --bootstrap "${BOOTSTRAP}" \
  --seed "${SEED}"

echo
echo "artifact: ${ARTIFACT}"
echo "report: ${REPORT}"
