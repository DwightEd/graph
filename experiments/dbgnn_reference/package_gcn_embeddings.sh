#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
SOURCE=${SOURCE:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare/gcn}
COMPARE=${COMPARE:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare}
OUT=${OUT:-${REPO}/experiments/dbgnn_reference/outputs/gcn_node_embeddings_qa.tar.gz}
INCLUDE_GRAPHS=${INCLUDE_GRAPHS:-0}

for SPLIT in calibration test; do
  if [ ! -f "${SOURCE}/${SPLIT}/index.npz" ]; then
    echo "Missing ${SOURCE}/${SPLIT}/index.npz"
    exit 1
  fi
done

STAGING=$(mktemp -d) || exit $?
trap 'rm -rf "${STAGING}"' EXIT
BUNDLE=${STAGING}/gcn_node_embeddings_qa

mkdir -p "${BUNDLE}/calibration" "${BUNDLE}/test" "${BUNDLE}/reference" || exit $?
cp "${SOURCE}/calibration/index.npz" "${BUNDLE}/calibration/index.npz" || exit $?
cp "${SOURCE}/test/index.npz" "${BUNDLE}/test/index.npz" || exit $?
cp \
  "${REPO}/experiments/dbgnn_reference/GCN_NODE_EMBEDDING_BUNDLE.md" \
  "${BUNDLE}/README.md" || exit $?

for FILE in detector.npz scores.npz; do
  if [ -f "${SOURCE}/${FILE}" ]; then
    cp "${SOURCE}/${FILE}" "${BUNDLE}/reference/${FILE}" || exit $?
  fi
done
if [ -f "${COMPARE}/diagnostics/report.json" ]; then
  cp \
    "${COMPARE}/diagnostics/report.json" \
    "${BUNDLE}/reference/diagnostics_report.json" || exit $?
fi

if [ "${INCLUDE_GRAPHS}" = "1" ]; then
  cp -a "${SOURCE}/calibration/graphs" "${BUNDLE}/calibration/graphs" || exit $?
  cp -a "${SOURCE}/test/graphs" "${BUNDLE}/test/graphs" || exit $?
fi

mkdir -p "$(dirname "${OUT}")" || exit $?
tar -czf "${OUT}" -C "${STAGING}" gcn_node_embeddings_qa || exit $?

echo "Created: ${OUT}"
du -h "${OUT}"
