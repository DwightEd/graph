#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
SPECTRAL_REFERENCE=${SPECTRAL_REFERENCE:-experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2/full/reference.npz}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

if [[ -n "${LIMIT:-}" ]]; then
  RUN_NAME="smoke_${LIMIT}"
  EXTRA=(--limit "$LIMIT")
  BOOTSTRAP_DEFAULT=100
else
  RUN_NAME=full
  EXTRA=()
  BOOTSTRAP_DEFAULT=1000
fi
OUT=${OUT:-experiments/rr_topology_dynamics/outputs/setwalk_coordination/$RUN_NAME}

if [[ ! -f "$SPECTRAL_REFERENCE" ]]; then
  cat >&2 <<EOF
missing frozen RR spectral reference: $SPECTRAL_REFERENCE
Run the RR spectral-subspace experiment first, or set SPECTRAL_REFERENCE to an existing reference.npz.
EOF
  exit 2
fi
if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi

mkdir -p "$OUT/evaluation"

echo "dataset_root=$ROOT"
echo "spectral_reference=$SPECTRAL_REFERENCE"
echo "output=$OUT"
echo "device=$DEVICE"
echo "lag_bins=${LAG_BINS:-8}"
echo "spectral_top_k=${SPECTRAL_TOP_K:-5}"
echo "position_bins=${POSITION_BINS:-8}"
echo "reference_per_sample=${REFERENCE_PER_SAMPLE:-16}"
echo "onset_window=${ONSET_WINDOW:-4}"
echo "bootstrap_replicates=${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}"

printf '\n[1/3] fit label-free topology feature reference on train\n'
"$PYTHON" -u -m experiments.rr_topology_dynamics.main fit \
  --train-split "$ROOT/train" \
  --spectral-reference "$SPECTRAL_REFERENCE" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --lag-bins "${LAG_BINS:-8}" \
  --spectral-top-k "${SPECTRAL_TOP_K:-5}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --position-bins "${POSITION_BINS:-8}" \
  --top-source-count "${TOP_SOURCE_COUNT:-8}" \
  --recent-lag-max "${RECENT_LAG_MAX:-4}" \
  --mid-lag-max "${MID_LAG_MAX:-16}" \
  --far-lag-fraction "${FAR_LAG_FRACTION:-0.5}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-16}" \
  --min-task-bin-rows "${MIN_TASK_BIN_ROWS:-8}" \
  --phase-bins "${PHASE_BINS:-10}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --seed "${SEED:-20260815}" \
  "${EXTRA[@]}"

printf '\n[2/3] freeze full test topology trajectories without labels\n'
"$PYTHON" -u -m experiments.rr_topology_dynamics.main score \
  --split-root "$ROOT/test" \
  --spectral-reference "$SPECTRAL_REFERENCE" \
  --topology-reference "$OUT/reference.npz" \
  --output "$OUT/test_features.npz" \
  --device "$DEVICE" \
  "${EXTRA[@]}"

printf '\n[3/3] post-hoc correct/error topology audit\n'
"$PYTHON" -u -m experiments.rr_topology_dynamics.main evaluate \
  --split-root "$ROOT/test" \
  --features "$OUT/test_features.npz" \
  --output-dir "$OUT/evaluation" \
  --device cpu \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --onset-window "${ONSET_WINDOW:-4}" \
  --phase-bins "${PHASE_BINS:-10}" \
  --seed "${SEED:-20260815}"

echo "done: $OUT/evaluation/report.json"
