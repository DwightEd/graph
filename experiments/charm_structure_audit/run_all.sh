#!/usr/bin/env bash
# CHARM audit, foreground. Existing CHECKPOINT skips all training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
BASE="/share/home/tm902089733300000/a903202310/lys"
OUTPUT="${OUTPUT:-outputs/charm_structure_audit}"
PREPARED="${PREPARED:-$OUTPUT/data}"
CHECKPOINT="${CHECKPOINT:-}"
PHASE="${PHASE:-all}"
read -r -a tasks <<< "${TASKS:-QA Summary Data2txt}"
read -r -a variants <<< "${VARIANTS:-charm_out charm_in node_only local_in rewire_in}"
read -r -a seeds <<< "${SEEDS:-0}"
"$PY" -m pytest tests/test_charm_structure_audit.py -q
if [[ "$PHASE" == all || "$PHASE" == prepare ]]; then
  splits=(train test)
  if [[ -n "$CHECKPOINT" ]]; then splits=("${SPLIT:-test}"); fi
  "$PY" -u -m experiments.charm_structure_audit.run prepare \
    --attention-root "${ATTENTION_ROOT:-$BASE/data/RAGTruth/attention/llama31_8b}" \
    --annotations "${ANNOTATIONS:-$BASE/data/RAGTruth/dataset/response.jsonl}" \
    --tokenizer "${TOKENIZER:-$BASE/models/Meta-Llama-3.1-8B-Instruct}" \
    --output "$PREPARED" --tasks "${tasks[@]}" --splits "${splits[@]}" \
    --generator "${GENERATOR:-llama-2-7b-chat}" --tau "${TAU:-0.05}" --limit "${LIMIT:-0}"
fi
common=(--prepared "$PREPARED" --output "$OUTPUT" --tasks "${tasks[@]}"
        --device "${DEVICE:-cuda}" --threads "${TORCH_THREADS:-1}"
        --bootstrap "${BOOTSTRAP:-200}" --edge-chunk "${EDGE_CHUNK:-4096}"
        --prefix-sites "${PREFIX_SITES:-8}")
if [[ "$PHASE" == all || "$PHASE" == fit || "$PHASE" == audit ]]; then
  if [[ -n "$CHECKPOINT" ]]; then
    "$PY" -u -m experiments.charm_structure_audit.run audit "${common[@]}" \
      --checkpoint "$CHECKPOINT" --variant "${CHECKPOINT_VARIANT:-charm_out}" \
      --threshold "${THRESHOLD:-0.5}" --split "${SPLIT:-test}" "$@"
  else
    if [[ "$PHASE" == audit ]]; then
      printf 'PHASE=audit requires the actual CHARM CHECKPOINT path.\n' >&2; exit 2
    fi
    printf '\nNo CHECKPOINT supplied: train supervised CHARM ablations; this is NOT the old run.\n'
    "$PY" -u -m experiments.charm_structure_audit.run fit "${common[@]}" \
      --variants "${variants[@]}" --seeds "${seeds[@]}" --epochs "${EPOCHS:-50}" \
      --batch-size "${BATCH_SIZE:-32}" --fpr "${FPR:-0.05}" "$@"
  fi
fi
