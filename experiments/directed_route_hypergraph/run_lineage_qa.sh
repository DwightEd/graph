#!/usr/bin/env bash

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
GRAPH_INDEX_ROOT=${GRAPH_INDEX_ROOT:-${REPO}/experiments/dbgnn_reference/outputs/qa_compare/gcn}
CALIBRATION_INDEX=${CALIBRATION_INDEX:-${GRAPH_INDEX_ROOT}/calibration/index.npz}
TEST_INDEX=${TEST_INDEX:-${GRAPH_INDEX_ROOT}/test/index.npz}
TEST_ROOT=${TEST_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
PYTHON=${PYTHON:-python}
SEED=${SEED:-20260827}
OUT=${OUT:-${REPO}/experiments/directed_route_hypergraph/outputs/qa/routing_lineage_gate_seed${SEED}}
CARRIER_REWIRE_PASSES=${CARRIER_REWIRE_PASSES:-4}
POSITION_BIN_WIDTH=${POSITION_BIN_WIDTH:-16}
MINIMUM_REFERENCE_SOURCES=${MINIMUM_REFERENCE_SOURCES:-10}
BOOTSTRAP=${BOOTSTRAP:-1000}
EVALUATE=${EVALUATE:-1}

cd "${REPO}" || exit 1
mkdir -p "${OUT}"

echo "[1/4] Trace calibration routing lineage without labels"
"${PYTHON}" -m experiments.directed_route_hypergraph.lineage_pipeline trace \
  --index "${CALIBRATION_INDEX}" \
  --output "${OUT}/calibration_trace.npz" \
  --seed "${SEED}" \
  --carrier-rewire-passes "${CARRIER_REWIRE_PASSES}" || exit $?

echo
echo "[2/4] Trace predecessor-aligned test routing lineage without labels"
"${PYTHON}" -m experiments.directed_route_hypergraph.lineage_pipeline trace \
  --index "${TEST_INDEX}" \
  --output "${OUT}/test_trace.npz" \
  --seed "${SEED}" \
  --carrier-rewire-passes "${CARRIER_REWIRE_PASSES}" || exit $?

echo
echo "[3/4] Freeze ordered and control scores before opening labels"
"${PYTHON}" -m experiments.directed_route_hypergraph.lineage_pipeline score \
  --calibration-trace "${OUT}/calibration_trace.npz" \
  --test-trace "${OUT}/test_trace.npz" \
  --output "${OUT}/scores.npz" \
  --position-bin-width "${POSITION_BIN_WIDTH}" \
  --minimum-reference-sources "${MINIMUM_REFERENCE_SOURCES}" || exit $?

if [ "${EVALUATE}" = "1" ]; then
  echo
  echo "[4/4] Evaluate the frozen mechanism scores"
  "${PYTHON}" -m experiments.directed_route_hypergraph.lineage_pipeline evaluate \
    --test-root "${TEST_ROOT}" \
    --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" \
    --bootstrap "${BOOTSTRAP}" --seed "${SEED}" || exit $?

  "${PYTHON}" - "${OUT}/evaluation.json" <<'PY' || exit $?
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)


def show(name, result):
    auroc = result.get("auroc")
    auprc = result.get("auprc")
    auroc_text = "NA" if auroc is None else f"{auroc:.6f}"
    auprc_text = "NA" if auprc is None else f"{auprc:.6f}"
    print(f"{name:28s} AUROC={auroc_text} AUPRC={auprc_text}")


print("\n=== Dataset ===")
print({
    "samples": report["samples"],
    "tokens_evaluated": report["tokens_evaluated"],
    "positive_tokens": report["positive_tokens"],
    "prevalence": report["prevalence"],
})

print("\n=== Routing drift ===")
for name, result in report["drift_detection"].items():
    show(name, result)

print("\n=== Routing dispersion ===")
for name, result in report["dispersion_detection"].items():
    show(name, result)

print("\n=== Position baselines ===")
for name, result in report["position_baselines"].items():
    show(name, result)

print("\n=== Mechanism audit ===")
print({
    "observability": report["observability"],
    "calibration": report["calibration_audit"],
    "carrier_rewire": report["carrier_rewire"],
    "dispersion": report["dispersion_audit"],
})
print(f"\nFull report: {sys.argv[1]}")
PY
else
  echo
  echo "[4/4] Labels remain closed"
fi

echo
echo "Finished: ${OUT}"
