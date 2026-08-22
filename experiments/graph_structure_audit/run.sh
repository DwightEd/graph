#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

ROOT=${ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
SPLIT=${SPLIT:-test}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-experiments/graph_structure_audit/outputs/audit_$RUN_ID}
PYTHON=${PYTHON:-python}

LIMIT=()
TASK=()
NO_PROGRESS=()
[[ -n "${LIMIT_SAMPLES:-}" ]] && LIMIT=(--limit "$LIMIT_SAMPLES")
[[ -n "${TASK_TYPE:-}" ]] && TASK=(--task-type "$TASK_TYPE")
[[ "${TQDM_DISABLE:-0}" == "1" ]] && NO_PROGRESS=(--no-progress)

mkdir -p "$OUT"

printf '\n[1/2] construct one causal multiplex graph per sample and freeze structural audits\n'
"$PYTHON" -u -m experiments.graph_structure_audit.main extract \
  --split-root "$ROOT/$SPLIT" \
  --output-dir "$OUT/audit" \
  --prompt-bins "${PROMPT_BINS:-16}" \
  --coalition-top-sources "${COALITION_TOP_SOURCES:-12}" \
  --source-mask-fraction "${SOURCE_MASK_FRACTION:-0.25}" \
  --channel-mask-fraction "${CHANNEL_MASK_FRACTION:-0.25}" \
  --seed "${SEED:-20260822}" \
  "${TASK[@]}" "${LIMIT[@]}" "${NO_PROGRESS[@]}"

printf '\n[2/2] unlock labels only for recoverability and structural comparison\n'
"$PYTHON" -u -m experiments.graph_structure_audit.main evaluate \
  --split-root "$ROOT/$SPLIT" \
  --tokens "$OUT/audit/tokens.npz" \
  --output-dir "$OUT/evaluation" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-500}" \
  --seed "${SEED:-20260822}"

printf '\nDone: %s\n' "$OUT"
