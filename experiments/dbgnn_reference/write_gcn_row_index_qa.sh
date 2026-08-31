#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
SOURCE=${SOURCE:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare/gcn}
BUNDLE=${BUNDLE:-${REPO}/experiments/dbgnn_reference/outputs/gcn_node_data_qa.npz}
OUTPUT=${OUTPUT:-${REPO}/experiments/dbgnn_reference/outputs/gcn_node_data_qa_rows.csv}
PYTHON=${PYTHON:-python}

cd "${REPO}" || exit $?

"${PYTHON}" -m experiments.dbgnn_reference.export_node_mapping \
  --bundle "${BUNDLE}" \
  --calibration-index "${SOURCE}/calibration/index.npz" \
  --test-index "${SOURCE}/test/index.npz" \
  --output "${OUTPUT}"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "ERROR: GCN row-index export failed with exit code ${STATUS}." >&2
  exit "${STATUS}"
fi

echo "row index: ${OUTPUT}"
