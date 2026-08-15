#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE=${BASE:-/share/home/tm902089733300000/a903202310/lys}
REPRESENTATION_ROOT=${1:?usage: bash attention_multiplex/run_signal_audit.sh REPRESENTATION_ROOT [ATTENTION_ROOT] [OUTPUT_DIR]}
DEFAULT_ATTENTION_ROOT="$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_ROOT=${2:-$DEFAULT_ATTENTION_ROOT}
OUTPUT_DIR=${3:-$REPRESENTATION_ROOT/signal_audit}

export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$OUTPUT_DIR/.matplotlib}
mkdir -p "$OUTPUT_DIR" "$MPLCONFIGDIR"

echo "representation_root=$REPRESENTATION_ROOT"
echo "attention_root=$ATTENTION_ROOT"
echo "output_dir=$OUTPUT_DIR"

python -m attention_multiplex.signal_audit \
  --representation-root "$REPRESENTATION_ROOT" \
  --attention-root "$ATTENTION_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --position-bins "${POSITION_BINS:-20}" \
  --shortlist "${SHORTLIST:-12}" \
  --bootstrap "${BOOTSTRAP:-200}"
