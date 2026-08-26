#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON=${PYTHON:-python}
OUT=${OUT:-"${ROOT}/experiments/grounded_route/outputs/graph_effectiveness"}
DEVICE=${DEVICE:-cuda}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-4}
BOOTSTRAP=${BOOTSTRAP:-2000}
SEEDS=${SEEDS:-"20260825 20260826 20260827"}

if [[ -z "${CALIBRATION_INDEX:-}" || -z "${GRAPH_INDEX:-}" || -z "${TEST_SPLIT:-}" ]]; then
  echo "CALIBRATION_INDEX, GRAPH_INDEX and TEST_SPLIT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}"

read -r -a SEED_ARGUMENTS <<< "${SEEDS}"
SCORE_ARGUMENT=()
if [[ -n "${UNSUPERVISED_SCORES:-}" ]]; then
  SCORE_ARGUMENT=(--scores "${UNSUPERVISED_SCORES}")
fi
read -r -a EXTRA_CONTROL_ARGUMENTS <<< "${CONTROL_ARGUMENTS:-}"

echo "[1/1] Verify embeddings, freeze node-only scores, then run diagnostics"
"${PYTHON}" -m experiments.grounded_route.graph_effectiveness.run audit \
  --calibration "${CALIBRATION_INDEX}" \
  --index "${GRAPH_INDEX}" \
  --test "${TEST_SPLIT}" \
  --output "${OUT}" \
  --device "${DEVICE}" \
  --folds "${FOLDS}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --bootstrap "${BOOTSTRAP}" \
  --seeds "${SEED_ARGUMENTS[@]}" \
  "${SCORE_ARGUMENT[@]}" \
  "${EXTRA_CONTROL_ARGUMENTS[@]}"

echo "Finished: ${OUT}"
