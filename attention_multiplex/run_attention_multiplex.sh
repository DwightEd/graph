#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE=${BASE:-/share/home/tm902089733300000/a903202310/lys}
DEFAULT_ATTENTION_ROOT="$BASE/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876"
ATTENTION_ROOT=${1:-${FORMAL_ROOT:-$DEFAULT_ATTENTION_ROOT}}
OUTPUT_ROOT=${2:-${OUTPUT_ROOT:-$BASE/data/feature_extraction/attention_multiplex/$(date -u +%Y%m%dT%H%M%SZ)}}
SPLITS=${SPLITS:-"train test"}
WORKERS=${WORKERS:-1}

# Each sample runs two CPU sparse SVDs.  With sample-level concurrency, keep
# native BLAS/OpenMP pools single-threaded to avoid workers multiplying their
# own thread pools.  CUDA_VISIBLE_DEVICES does not accelerate this CPU path.
if (( WORKERS > 1 )); then
  export OMP_NUM_THREADS=${BLAS_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${BLAS_THREADS:-1}
  export MKL_NUM_THREADS=${BLAS_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${BLAS_THREADS:-1}
fi

export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_ROOT"

echo "attention_root=$ATTENTION_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "splits=$SPLITS"
echo "rank=${RANK:-16}"
echo "workers=$WORKERS"
echo "resume=${RESUME:-0}"

for split in $SPLITS; do
  split_root="$ATTENTION_ROOT/$split"
  if [[ ! -f "$split_root/manifest.json" ]]; then
    if [[ -f "$ATTENTION_ROOT/manifest.json" && "$SPLITS" == "$split" ]]; then
      split_root="$ATTENTION_ROOT"
    else
      echo "missing split manifest: $split_root/manifest.json" >&2
      exit 1
    fi
  fi

  args=(
    --attention-split "$split_root"
    --output-dir "$OUTPUT_ROOT/$split"
    --device "${DEVICE:-cpu}"
    --rank "${RANK:-16}"
    --block-rows "${BLOCK_ROWS:-4096}"
    --seed "${SEED:-20260815}"
    --workers "$WORKERS"
    --checkpoint-every "${CHECKPOINT_EVERY:-10}"
  )
  if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "${INCLUDE_DIAGONAL:-1}" == "0" ]]; then
    args+=(--exclude-diagonal)
  fi
  if [[ "${RESUME:-0}" == "1" ]]; then
    args+=(--resume)
  fi

  echo "processing_split=$split"
  python -m attention_multiplex.cli "${args[@]}"
done

echo "complete_output=$OUTPUT_ROOT"
