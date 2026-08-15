#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the current label-free causal-topology experiment.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-/share/home/tm902089733300000/a903202310/lys}"

export DATA_ROOT="${DATA_ROOT:-${ROOT:-$BASE/data/RAGTruth/model_traces/llama31_8b}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/causal_topology/$(date -u +%Y%m%dT%H%M%SZ)}}"
export BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-${BOOTSTRAP:-200}}"

exec bash "$PROJECT_ROOT/run_token_representation.sh"
