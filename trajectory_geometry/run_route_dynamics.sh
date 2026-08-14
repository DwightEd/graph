#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-/share/home/tm902089733300000/a903202310/lys}
DEFAULT_ATTENTION_ROOT="$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_CACHE_ROOT=${1:-${FORMAL_ROOT:-$DEFAULT_ATTENTION_ROOT}}
OUTPUT_DIR=${2:-$BASE/data/feature_extraction/trajectory_geometry/$(date -u +%Y%m%dT%H%M%SZ)}
SPLIT=${3:-${SPLIT:-train}}

cd "$SCRIPT_DIR"
echo "attention_root=$ATTENTION_CACHE_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "split=$SPLIT"

ARGS=(
  --attention-root "$ATTENTION_CACHE_ROOT"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --embedding-dim "${EMBEDDING_DIM:-256}"
  --csr-row-block "${CSR_ROW_BLOCK:-4096}"
)
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

python -m trajectory_geometry.cli extract "${ARGS[@]}"
