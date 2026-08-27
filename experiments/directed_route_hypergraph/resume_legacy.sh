#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEGACY_COMMIT=${LEGACY_COMMIT:-80acf557180132c95f4daac4417aa17219426a90}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TASK=${TASK:-QA}
SEED=${SEED:-20260827}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EVALUATE=${EVALUATE:-1}
TRAIN_SPLIT=${TRAIN_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
CHECKPOINT=${CHECKPOINT:-${ROOT}/experiments/directed_route_hypergraph/outputs/${TASK,,}/real_seed${SEED}/model.pt}
OUT=${OUT:-$(dirname "${CHECKPOINT}")/legacy_recovery}
TEST_LIMIT=${TEST_LIMIT:-}

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [ ! -d "${TRAIN_SPLIT}" ] || [ ! -d "${TEST_SPLIT}" ]; then
  echo "TRAIN_SPLIT and TEST_SPLIT must point to existing dataset splits." >&2
  exit 1
fi
if ! git -C "${ROOT}" cat-file -e "${LEGACY_COMMIT}^{commit}" 2>/dev/null; then
  echo "Legacy commit is unavailable locally: ${LEGACY_COMMIT}" >&2
  echo "Run 'git fetch origin' and retry." >&2
  exit 1
fi

WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/directed-route-legacy.XXXXXX")"
cleanup() {
  git -C "${ROOT}" worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || \
    rm -rf "${WORKTREE}"
}
trap cleanup EXIT INT TERM

git -C "${ROOT}" worktree add --detach --quiet "${WORKTREE}" "${LEGACY_COMMIT}"
mkdir -p "${OUT}"
printf '%s\n' "${LEGACY_COMMIT}" > "${OUT}/legacy_source_commit.txt"

run_legacy() {
  (
    cd "${WORKTREE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON}" "$@"
  )
}

# Fail before expensive encoding unless this is the exact learned-head-transition checkpoint.
run_legacy - "${CHECKPOINT}" <<'PY'
from pathlib import Path
import sys
import torch

from experiments.directed_route_hypergraph.pipeline import restore_model

path = Path(sys.argv[1])
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
config = dict(checkpoint.get("model_config", {}))
if "head_transition_identity_bias" not in config:
    raise SystemExit(
        "checkpoint is not the legacy head-transition model; do not use this recovery script"
    )
checkpoint, model = restore_model(path, "cpu")
actual = sum(parameter.numel() for parameter in model.parameters())
recorded = int(checkpoint.get("parameter_count", actual))
if actual != recorded:
    raise SystemExit(
        f"checkpoint parameter count mismatch: recorded={recorded}, restored={actual}"
    )
print(f"legacy checkpoint verified: {actual} parameters")
PY

TEST_LIMIT_ARGUMENT=()
[ -n "${TEST_LIMIT}" ] && TEST_LIMIT_ARGUMENT=(--limit "${TEST_LIMIT}")

printf '\n[2/5] Export calibration embeddings with the exact legacy source\n'
run_legacy -m experiments.directed_route_hypergraph.run encode \
  --data "${TRAIN_SPLIT}" --checkpoint "${CHECKPOINT}" \
  --output "${OUT}/calibration" --scope calibration \
  --task "${TASK}" --device "${DEVICE}"

printf '\n[3/5] Export test embeddings with the exact legacy source\n'
run_legacy -m experiments.directed_route_hypergraph.run encode \
  --data "${TEST_SPLIT}" --checkpoint "${CHECKPOINT}" \
  --output "${OUT}/test" --scope all --task "${TASK}" \
  --device "${DEVICE}" "${TEST_LIMIT_ARGUMENT[@]}"

printf '\n[4/5] Fit node-only PCA-kNN and freeze scores\n'
run_legacy -m experiments.directed_route_hypergraph.run detect \
  --calibration "${OUT}/calibration/index.npz" \
  --test "${OUT}/test/index.npz" \
  --reference "${OUT}/detector.npz" --scores "${OUT}/scores.npz" \
  --seed "${SEED}"

if [ "${EVALUATE}" = "1" ]; then
  printf '\n[5/5] Evaluate frozen token scores\n'
  run_legacy -m experiments.directed_route_hypergraph.run evaluate \
    --test "${TEST_SPLIT}" --scores "${OUT}/scores.npz" \
    --output "${OUT}/evaluation.json" --seed "${SEED}"
else
  printf '\n[5/5] Labels remain closed\n'
fi

printf '\nRecovered legacy run: %s\n' "${OUT}"
