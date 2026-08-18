#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -n "${LIMIT:-}" ]]; then
  RUN_NAME="smoke_${LIMIT}_${RUN_ID}"
  EXTRA=(--limit "$LIMIT")
  BOOTSTRAP_DEFAULT=50
else
  RUN_NAME="full_${RUN_ID}"
  EXTRA=()
  BOOTSTRAP_DEFAULT=200
fi
OUT=${OUT:-experiments/causal_attention_setwalk/outputs/$RUN_NAME}

if [[ ! -f "$ROOT/train/manifest.json" || ! -f "$ROOT/test/manifest.json" ]]; then
  echo "dataset root does not contain train/test manifests: $ROOT" >&2
  exit 2
fi

mkdir -p "$OUT/evaluation"
echo "method=causal_attention_setwalk"
echo "dataset_root=$ROOT"
echo "output=$OUT"
echo "device=$DEVICE"
echo "fourier_features=${FOURIER_FEATURES:-8}"
echo "dct_components=${DCT_COMPONENTS:-3}"
echo "recent_lag_max=${RECENT_LAG_MAX:-4}"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  printf '\n[preflight] causal SetWalk contracts\n'
  "$PYTHON" -m unittest \
    experiments.causal_attention_setwalk.tests.test_representation \
    experiments.causal_attention_setwalk.tests.test_model
fi

printf '\n[1/3] fit unlabeled train SetWalk references\n'
"$PYTHON" -u -m experiments.causal_attention_setwalk.main fit \
  --train-split "$ROOT/train" \
  --output "$OUT/reference.npz" \
  --device "$DEVICE" \
  --fourier-features "${FOURIER_FEATURES:-8}" \
  --dct-components "${DCT_COMPONENTS:-3}" \
  --recent-lag-max "${RECENT_LAG_MAX:-4}" \
  --block-rows "${BLOCK_ROWS:-8192}" \
  --seed "${SEED:-20260818}" \
  --reference-per-sample "${REFERENCE_PER_SAMPLE:-8}" \
  --position-bins "${POSITION_BINS:-8}" \
  --min-task-bin-rows "${MIN_TASK_BIN_ROWS:-8}" \
  --trim-fraction "${TRIM_FRACTION:-0.90}" \
  "${EXTRA[@]}"

printf '\n[2/3] freeze test token embeddings and ablations without labels\n'
"$PYTHON" -u -m experiments.causal_attention_setwalk.main score \
  --split-root "$ROOT/test" \
  --reference "$OUT/reference.npz" \
  --output "$OUT/nodes.npz" \
  --device "$DEVICE" \
  "${EXTRA[@]}"

printf '\n[3/3] post-hoc mechanism validation\n'
"$PYTHON" -u -m experiments.causal_attention_setwalk.main evaluate \
  --split-root "$ROOT/test" \
  --scores "$OUT/nodes.npz" \
  --output-dir "$OUT/evaluation" \
  --device cpu \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-$BOOTSTRAP_DEFAULT}" \
  --seed "${SEED:-20260818}"

echo "complete_output=$OUT"
echo "node_representations=$OUT/nodes.npz"
echo "evaluation=$OUT/evaluation/evaluation.json"
echo "metrics=$OUT/evaluation/metrics.csv"

