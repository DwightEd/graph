#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  cat <<'HELP'
Run prepare -> extract -> detect -> evaluate using an existing GPU Python environment.

Environment parameters:
  MODEL_PATH        Local Llama checkpoint (default: lys/models/Meta-Llama-3.1-8B-Instruct)
  RAGTRUTH_DIR      Dataset containing source_info.jsonl and response.jsonl
  PYTHON_BIN        Python executable (default: python)
  OUTPUT_DIR        New output directory (default: outputs/routes_<timestamp>_<pid>)
  DEVICE / DTYPE    Defaults: cuda:0 / bfloat16
  TASK / GENERATOR  Defaults: QA / llama-2-7b-chat (observer replay)
  MAX_SOURCES       Default: 256, half reference and half test
  END_LAYERS        Space-separated zero-based layers; empty means final layer
  DEPTHS            Default: "1 2"
  CANDIDATES        Default: 2
  MAX_TOKENS        Default: 2048; no silent truncation
  MAX_ATTENTION_MB  Default: 16384; full attention storage estimate, not total VRAM
  NEIGHBORS         Default: 3 distinct reference sources per stratum
  PER_SOURCE        Default: 8 reference examples per source per stratum
  BOOTSTRAP / SEED  Defaults: 200 / 20260911

No package installation, checkpoint download, or threshold tuning is performed.
Scores and metrics are detection baselines; there is no reanchor event detector yet.
HELP
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf 'Use environment parameters; see --help.\n' >&2
  exit 2
fi

python_bin=${PYTHON_BIN:-python}
model=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
dataset=${RAGTRUTH_DIR:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset}
output=${OUTPUT_DIR:-"$repo_root/outputs/routes_$(date +%Y%m%d_%H%M%S)_$$"}
case "$output" in
  /*|[A-Za-z]:/*) ;;
  *) output="$repo_root/$output" ;;
esac
log="${output}.log"
device=${DEVICE:-cuda:0}
dtype=${DTYPE:-bfloat16}

if [[ ! -f "$model/config.json" || ! -f "$dataset/source_info.jsonl" || ! -f "$dataset/response.jsonl" ]]; then
  printf 'Local model config and both RAGTruth JSONL files must exist.\n' >&2
  exit 2
fi
if [[ -e "$output" || -e "$log" ]]; then
  printf 'Refusing existing output/log: %s\n' "$output" >&2
  exit 2
fi
mkdir -p "$output"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

read -r -a depths <<< "${DEPTHS:-1 2}"
read -r -a end_layers <<< "${END_LAYERS:-}"
extract_args=(
  extract --input "$output/input.jsonl" --output "$output/features" --model "$model"
  --device "$device" --dtype "$dtype" --candidates "${CANDIDATES:-2}"
  --depths "${depths[@]}" --max-tokens "${MAX_TOKENS:-2048}"
  --max-attention-mb "${MAX_ATTENTION_MB:-16384}"
)
if [[ ${#end_layers[@]} -gt 0 ]]; then
  extract_args+=(--end-layers "${end_layers[@]}")
fi

run() {
  printf 'Output: %s\nProbe: %s\nDevice: %s (%s)\n' "$output" "$model" "$device" "$dtype"
  "$python_bin" - "$device" "$dtype" <<'PY'
import sys
import torch
import transformers
import sklearn
import numpy

print(f"torch={torch.__version__}; transformers={transformers.__version__}; numpy={numpy.__version__}; sklearn={sklearn.__version__}")
device = torch.device(sys.argv[1])
if device.type == 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment')
    with torch.cuda.device(device):
        if sys.argv[2] == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Selected GPU does not support bfloat16; set DTYPE=float16')
        free, total = torch.cuda.mem_get_info()
        print(f"GPU={torch.cuda.get_device_name()}; free={free / 2**30:.1f} GiB; total={total / 2**30:.1f} GiB")
PY
  printf '\n[1/4] Prepare without annotation-based selection\n'
  "$python_bin" -u main.py prepare --dataset "$dataset" --task "${TASK:-QA}" \
    --generator "${GENERATOR:-llama-2-7b-chat}" --max-sources "${MAX_SOURCES:-256}" \
    --seed "${SEED:-20260911}" --output "$output/input.jsonl"
  printf '\n[2/4] Frozen causal graph extraction\n'
  "$python_bin" -u main.py "${extract_args[@]}"
  printf '\n[3/4] Source-disjoint reference and frozen scores\n'
  "$python_bin" -u main.py detect --features "$output/features" --output "$output/detection" \
    --neighbors "${NEIGHBORS:-3}" --per-source "${PER_SOURCE:-8}"
  printf '\n[4/4] Evaluation-only annotation join\n'
  "$python_bin" -u main.py evaluate --scores "$output/detection/scores.jsonl" \
    --labels "$dataset/response.jsonl" --output "$output/evaluation.json" \
    --bootstrap "${BOOTSTRAP:-200}" --seed "${SEED:-20260911}"
  printf 'All four stages completed; detection effectiveness must be read from evaluation.json.\n' > "$output/COMPLETE"
  printf '\nCompleted: %s/evaluation.json\n' "$output"
}

trap 'status=$?; printf "FAILED (exit %s); keep partial artifacts and inspect %s\n" "$status" "$log" >&2; exit "$status"' ERR
run 2>&1 | tee "$log"
