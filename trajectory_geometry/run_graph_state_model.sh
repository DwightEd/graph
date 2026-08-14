#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-/share/home/tm902089733300000/a903202310/lys}
DEFAULT_ATTENTION_ROOT="$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_ROOT=${ATTENTION_ROOT:-$DEFAULT_ATTENTION_ROOT}
HIDDEN_ROOT=${HIDDEN_ROOT:-${1:-}}
OUTPUT_DIR=${OUTPUT_DIR:-${2:-$BASE/data/feature_extraction/graph_state_model/$(date -u +%Y%m%dT%H%M%SZ)}}
PYTHON=${PYTHON:-$BASE/conda_envs/research/bin/python}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}.log}

if [[ -z "$HIDDEN_ROOT" ]]; then
  echo "usage: HIDDEN_ROOT=/actual/hidden_cache bash trajectory_geometry/run_graph_state_model.sh" >&2
  echo "or: bash trajectory_geometry/run_graph_state_model.sh /actual/hidden_cache [output_dir]" >&2
  exit 2
fi

cd "$SCRIPT_DIR"
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
export PYTHONUNBUFFERED=1

printf 'attention_root=%s\nhidden_root=%s\noutput_dir=%s\n' \
  "$ATTENTION_ROOT" "$HIDDEN_ROOT" "$OUTPUT_DIR"

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
  "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
fi

ARGS=(
  --attention-root "$ATTENTION_ROOT"
  --hidden-root "$HIDDEN_ROOT"
  --output-dir "$OUTPUT_DIR"
  --projection-dim "${PROJECTION_DIM:-16}"
  --projection-reference-rows "${PROJECTION_REFERENCE_ROWS:-12000}"
  --head-components "${HEAD_COMPONENTS:-8}"
  --fit-tokens-per-layer "${FIT_TOKENS_PER_LAYER:-4096}"
  --fit-fraction "${FIT_FRACTION:-0.8}"
  --trim-fraction "${TRIM_FRACTION:-0.9}"
  --ridge "${RIDGE:-0.01}"
  --residual-shrinkage "${RESIDUAL_SHRINKAGE:-0.1}"
  --minimum-relative-graph-gain "${MINIMUM_RELATIVE_GRAPH_GAIN:-0.01}"
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-1000}"
  --dct-components "${DCT_COMPONENTS:-8}"
  --prompt-rewire-bins "${PROMPT_REWIRE_BINS:-8}"
  --csr-row-block "${CSR_ROW_BLOCK:-4096}"
  --seed "${SEED:-20260815}"
)
if [[ -n "${LIMIT_TRAIN:-}" ]]; then
  ARGS+=(--limit-train "$LIMIT_TRAIN")
fi
if [[ -n "${LIMIT_TEST:-}" ]]; then
  ARGS+=(--limit-test "$LIMIT_TEST")
fi
if [[ "${SAVE_TRAIN_EMBEDDINGS:-1}" != "1" ]]; then
  ARGS+=(--skip-train-embeddings)
fi

"$PYTHON" -u -m trajectory_geometry.cli fit-state-model "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

printf 'complete_output=%s\nmodel=%s\nmanifest=%s\nrun_log=%s\n' \
  "$OUTPUT_DIR" "$OUTPUT_DIR/graph_state_model.npz" "$OUTPUT_DIR/manifest.json" "$LOG_FILE"
