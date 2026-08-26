#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-"${ROOT}/experiments/grounded_route/outputs/evaluation"}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-20}
BOOTSTRAP=${BOOTSTRAP:-1000}
SEEDS=${SEEDS:-"20260825 20260826 20260827"}

if [ -z "${CALIBRATION_INDEX:-}" ] || [ -z "${TEST_INDEX:-}" ] || [ -z "${TEST_ROOT:-}" ]; then
  echo "CALIBRATION_INDEX, TEST_INDEX and TEST_ROOT must be set."
  exit 1
fi

mkdir -p "${OUT}"
cd "${ROOT}" || exit 1

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a CONTROL_ARGS <<< "${CONTROL_ARGUMENTS:-}"

"${PYTHON}" -m experiments.grounded_route.evaluation.run \
  --calibration "${CALIBRATION_INDEX}" \
  --test-index "${TEST_INDEX}" \
  --test-root "${TEST_ROOT}" \
  --output "${OUT}" \
  --device "${DEVICE}" \
  --folds "${FOLDS}" \
  --epochs "${EPOCHS}" \
  --bootstrap "${BOOTSTRAP}" \
  --seeds "${SEED_ARGS[@]}" \
  "${CONTROL_ARGS[@]}" || exit $?

printf '\nFinished: %s\n' "${OUT}/report.json"
