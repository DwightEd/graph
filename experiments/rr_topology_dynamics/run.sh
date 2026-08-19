#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

LOCAL_DATA_ROOT="$PROJECT_ROOT/../data/RAGTruth/llama31_8b"
LOCAL_PYTHON="$PROJECT_ROOT/../../.audit_envs/llm_state_lab_py311/Scripts/python.exe"
REMOTE_DATA_ROOT=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876

if [[ -z "${ROOT:-}" ]]; then
  if [[ -f "$LOCAL_DATA_ROOT/train/manifest.json" && -f "$LOCAL_DATA_ROOT/test/manifest.json" ]]; then
    ROOT=$LOCAL_DATA_ROOT
  else
    ROOT=$REMOTE_DATA_ROOT
  fi
fi

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -f "$LOCAL_PYTHON" ]]; then
    PYTHON=$LOCAL_PYTHON
  else
    PYTHON=python
  fi
fi

if [[ "$PYTHON" == */* ]]; then
  [[ -f "$PYTHON" ]] || { echo "python executable does not exist: $PYTHON" >&2; exit 2; }
else
  command -v "$PYTHON" >/dev/null 2>&1 || { echo "python executable is not on PATH: $PYTHON" >&2; exit 2; }
fi

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  echo "Set ROOT to the RAGTruth llama31_8b directory." >&2
  exit 2
fi

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE=$(
    "$PYTHON" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")'
  )
fi

if [[ -n "${LIMIT:-}" ]]; then
  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "LIMIT must be a positive integer" >&2; exit 2; }
  RUN_MODE="smoke_${LIMIT}"
  EXTRA=(--limit "$LIMIT")
  SPECTRAL_LIMIT=$LIMIT
  if ((SPECTRAL_LIMIT < 32)); then
    SPECTRAL_LIMIT=32
  fi
  SPECTRAL_EXTRA=(--limit "$SPECTRAL_LIMIT")
  BOOTSTRAP_DEFAULT=100
else
  RUN_MODE=full
  EXTRA=()
  SPECTRAL_EXTRA=()
  BOOTSTRAP_DEFAULT=1000
fi

RUN_ID=${RUN_ID:-${RUN_MODE}_$(date +%Y%m%d_%H%M%S)}
SPECTRAL_REFERENCE=${SPECTRAL_REFERENCE:-experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2/$RUN_MODE/reference.npz}
OUT=${OUT:-experiments/rr_topology_dynamics/outputs/prompt_attractor/$RUN_ID}
LOG_DIR="$OUT/logs"
mkdir -p "$(dirname "$SPECTRAL_REFERENCE")" "$OUT/evaluation" "$LOG_DIR"

run_stage() {
  local name=$1
  shift
  local log="$LOG_DIR/${name}.log"
  printf '\n[%s] starting; log=%s\n' "$name" "$log"
  set +e
  "$@" 2>&1 | tee "$log"
  local -a statuses=("${PIPESTATUS[@]}")
  set -e
  if ((statuses[0] != 0)); then
    printf '[%s] failed with exit code %s; see %s\n' "$name" "${statuses[0]}" "$log" >&2
    return "${statuses[0]}"
  fi
  if ((statuses[1] != 0)); then
    printf '[%s] completed but its log could not be written: %s\n' "$name" "$log" >&2
    return "${statuses[1]}"
  fi
  printf '[%s] completed\n' "$name"
}

require_file() {
  local path=$1
  local stage=$2
  if [[ ! -f "$path" ]]; then
    echo "[$stage] command returned success but did not create: $path" >&2
    exit 1
  fi
}

echo "experiment=rr_topology_dynamics (prompt-grounded attractor audit)"
echo "dataset_root=$ROOT"
echo "python=$PYTHON"
echo "device=$DEVICE"
echo "spectral_reference=$SPECTRAL_REFERENCE"
echo "output=$OUT"
echo "limit=${LIMIT:-all}"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  run_stage preflight_tests \
    "$PYTHON" -m pytest -q \
    experiments/rr_topology_dynamics/tests \
    tests/test_spectral_feasibility.py
fi

if [[ ! -f "$SPECTRAL_REFERENCE" ]]; then
  run_stage spectral_fit \
    "$PYTHON" -u -m experiments.spectral_feasibility.main fit \
    --train-split "$ROOT/train" \
    --output "$SPECTRAL_REFERENCE" \
    --device "$DEVICE" \
    --top-k "${SPECTRAL_TOP_K:-5}" \
    --position-bins "${SPECTRAL_POSITION_BINS:-4}" \
    --pca-dim "${PCA_DIM:-32}" \
    --reference-per-sample "${SPECTRAL_REFERENCE_PER_SAMPLE:-6}" \
    --trim-fraction "${TRIM_FRACTION:-0.90}" \
    --calibration-fraction "${CALIBRATION_FRACTION:-0.25}" \
    --split-seed "${SEED:-20260815}" \
    --channel-tail-fraction "${CHANNEL_TAIL_FRACTION:-0.05}" \
    --attribution-topk "${ATTRIBUTION_TOPK:-8}" \
    --block-rows "${BLOCK_ROWS:-8192}" \
    "${SPECTRAL_EXTRA[@]}"
  require_file "$SPECTRAL_REFERENCE" spectral_fit
else
  echo "[spectral_fit] reusing existing reference: $SPECTRAL_REFERENCE"
fi

run_stage topology_fit \
  "$PYTHON" -u -m experiments.rr_topology_dynamics.main fit \
  --train-split "$ROOT/train" \
  --spectral-reference "$SPECTRAL_REFERENCE" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --spectral-top-k "${SPECTRAL_TOP_K:-5}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --position-bins "${POSITION_BINS:-8}" \
  --recent-lag-max "${RECENT_LAG_MAX:-4}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-16}" \
  --min-task-bin-rows "${MIN_TASK_BIN_ROWS:-8}" \
  --phase-bins "${PHASE_BINS:-10}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --seed "${SEED:-20260815}" \
  "${EXTRA[@]}"
require_file "$OUT/reference.npz" topology_fit

run_stage topology_score \
  "$PYTHON" -u -m experiments.rr_topology_dynamics.main score \
  --split-root "$ROOT/test" \
  --spectral-reference "$SPECTRAL_REFERENCE" \
  --topology-reference "$OUT/reference.npz" \
  --output "$OUT/test_features.npz" \
  --device "$DEVICE" \
  "${EXTRA[@]}"
require_file "$OUT/test_features.npz" topology_score

run_stage topology_evaluate \
  "$PYTHON" -u -m experiments.rr_topology_dynamics.main evaluate \
  --split-root "$ROOT/test" \
  --features "$OUT/test_features.npz" \
  --output-dir "$OUT/evaluation" \
  --device cpu \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --phase-bins "${PHASE_BINS:-10}" \
  --seed "${SEED:-20260815}"
require_file "$OUT/evaluation/report.json" topology_evaluate

echo
echo "done: $OUT/evaluation/report.json"
