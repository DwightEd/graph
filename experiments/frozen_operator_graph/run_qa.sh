#!/usr/bin/env bash

# One-command launcher for the exact, label-free QA operator graph.
# Run from any directory with:
#   conda run -n research bash experiments/frozen_operator_graph/run_qa.sh
#
# The raw RAGTruth JSONL is recorded and hashed as source provenance.  Exact
# graph construction itself consumes the token-aligned formal attention cache;
# reparsing response.jsonl here would risk changing the cached token alignment.

fail_run() {
  local message=$1
  local code=${2:-1}
  echo >&2
  echo "ERROR: ${message}" >&2
  echo "Stopped immediately; no approximate or reduced-data fallback was used." >&2
  exit "${code}"
}

run_stage() {
  local stage_name=$1
  local status
  shift
  "$@"
  status=$?
  if [ "${status}" -ne 0 ]; then
    fail_run "${stage_name} failed with exit code ${status}." "${status}"
  fi
}

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
RAGTRUTH_RESPONSE_JSON=${RAGTRUTH_RESPONSE_JSON:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl}
CACHE_PROJECT_ROOT=${CACHE_PROJECT_ROOT:-/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph}
FORMAL_CACHE_ROOT=${FORMAL_CACHE_ROOT:-${CACHE_PROJECT_ROOT}/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876}
SPLIT_ROOT=${SPLIT_ROOT:-${FORMAL_CACHE_ROOT}/test}

PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda:0}
MODEL_DTYPE=${MODEL_DTYPE:-auto}
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/frozen_operator_graph/outputs/qa/${MODEL_TAG}_full}

ROUTE_MASS_RETENTION=${ROUTE_MASS_RETENTION:-1.0}
VALUE_ENERGY_RETENTION=${VALUE_ENERGY_RETENTION:-1.0}
MINIMUM_ROLE_EDGES=${MINIMUM_ROLE_EDGES:-1}
CONSERVATION_ATOL=${CONSERVATION_ATOL:-0.005}
CONSERVATION_RTOL=${CONSERVATION_RTOL:-0.005}
CACHE_BINDING_ATOL=${CACHE_BINDING_ATOL:-0.005}
OUTPUT_DTYPE=${OUTPUT_DTYPE:-float32}
VERIFY_HASHES=${VERIFY_HASHES:-1}
OVERWRITE=${OVERWRITE:-0}
ALLOW_REMOTE_FILES=${ALLOW_REMOTE_FILES:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}
LIMIT=${LIMIT:-}
REVISION=${REVISION:-}

[ -d "${REPO}" ] || fail_run "repository directory is missing: ${REPO}"
[ -d "${MODEL_PATH}" ] || fail_run "model directory is missing: ${MODEL_PATH}"
[ -f "${RAGTRUTH_RESPONSE_JSON}" ] || fail_run "RAGTruth response JSONL is missing: ${RAGTRUTH_RESPONSE_JSON}"
[ -f "${SPLIT_ROOT}/manifest.json" ] || fail_run \
  "exact formal attention split is missing: ${SPLIT_ROOT}. The raw response.jsonl alone cannot reproduce the cached token/head alignment required by this method."

run_stage "enter repository ${REPO}" cd "${REPO}"

resolve_cache_provenance() {
  "${PYTHON}" - \
    "${SPLIT_ROOT}/manifest.json" \
    "${MODEL_PATH}" \
    "${RAGTRUTH_RESPONSE_JSON}" \
    "${MODEL_DTYPE}" <<'PY'
import json
from pathlib import Path
import sys

import torch
import transformers

manifest_path = Path(sys.argv[1]).expanduser().resolve()
model_path = Path(sys.argv[2]).expanduser().resolve()
source_json = Path(sys.argv[3]).expanduser().resolve()
requested_dtype = sys.argv[4].strip().lower()

with manifest_path.open(encoding="utf-8") as handle:
    manifest = json.load(handle)
spec = manifest.get("attention_cache_spec")
if not isinstance(spec, dict):
    raise ValueError(
        f"{manifest_path} has no attention_cache_spec; use the original formal "
        "fresh_attention cache, not a provenance-truncated copy"
    )
if str(manifest.get("state", "")).casefold() != "complete":
    raise ValueError(f"attention cache is not complete: {manifest_path}")
if str(spec.get("attn_implementation", "")).casefold() != "eager":
    raise ValueError("the formal cache was not produced with eager attention")

recorded_model = spec.get("model_path")
if recorded_model:
    recorded_path = Path(str(recorded_model)).expanduser()
    if recorded_path.exists() and recorded_path.resolve() != model_path:
        raise ValueError(
            "MODEL_PATH differs from the checkpoint recorded by the cache: "
            f"cache={recorded_path.resolve()}, requested={model_path}"
        )

recorded_transformers = spec.get("transformers_version")
recorded_torch = spec.get("torch_version")
if recorded_transformers and str(recorded_transformers) != str(transformers.__version__):
    raise ValueError(
        "Transformers version differs from cache extraction: "
        f"cache={recorded_transformers}, runtime={transformers.__version__}"
    )
if recorded_torch and str(recorded_torch) != str(torch.__version__):
    raise ValueError(
        "PyTorch version/build differs from cache extraction: "
        f"cache={recorded_torch}, runtime={torch.__version__}"
    )

raw_dtype = str(spec.get("dtype", "")).casefold()
if "bfloat16" in raw_dtype:
    inferred_dtype = "bfloat16"
elif "float16" in raw_dtype or raw_dtype in {"half", "torch.half"}:
    inferred_dtype = "float16"
elif "float32" in raw_dtype or raw_dtype in {"float", "torch.float"}:
    inferred_dtype = "float32"
else:
    raise ValueError(
        "cannot infer exact replay dtype from attention_cache_spec.dtype: "
        f"{spec.get('dtype')!r}"
    )

if requested_dtype == "auto":
    selected_dtype = inferred_dtype
else:
    if requested_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("MODEL_DTYPE must be auto, float32, float16, or bfloat16")
    if requested_dtype != inferred_dtype:
        raise ValueError(
            "explicit MODEL_DTYPE disagrees with the formal cache: "
            f"cache={inferred_dtype}, requested={requested_dtype}"
        )
    selected_dtype = requested_dtype

print(f"MODEL_DTYPE={selected_dtype}")
print(f"CACHE_MANIFEST={manifest_path}")
print(f"CACHE_SAMPLES={manifest.get('count')}")
print(f"CACHE_MODEL={spec.get('model_path')}")
print(f"CACHE_COMPUTE_DTYPE={spec.get('dtype')}")
print(f"CACHE_STORAGE_DTYPE={spec.get('cache_dtype')}")
print(f"CACHE_TRANSFORMERS={recorded_transformers}")
print(f"CACHE_TORCH={recorded_torch}")
print(f"RAW_SOURCE_JSON={source_json}")
PY
}

PROVENANCE_OUTPUT=$(resolve_cache_provenance) || fail_run "cache/model/runtime provenance preflight failed"
echo "${PROVENANCE_OUTPUT}"
RESOLVED_MODEL_DTYPE=$(printf '%s\n' "${PROVENANCE_OUTPUT}" | sed -n 's/^MODEL_DTYPE=//p' | tail -n 1)
[ -n "${RESOLVED_MODEL_DTYPE}" ] || fail_run "failed to resolve MODEL_DTYPE from formal cache provenance"

if [ -d "${OUT}" ] && [ -n "$(find "${OUT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && [ "${OVERWRITE}" != "1" ]; then
  fail_run "output directory is not empty: ${OUT}. Set OVERWRITE=1 only when replacement is intentional."
fi

run_stage "create output parent" mkdir -p "$(dirname "${OUT}")"

echo
echo "=== FROZEN HYPERNETWORK OPERATOR GRAPH ==="
echo "REPO: ${REPO}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "RAGTRUTH_RESPONSE_JSON: ${RAGTRUTH_RESPONSE_JSON}"
echo "SPLIT_ROOT: ${SPLIT_ROOT}"
echo "OUT: ${OUT}"
echo "DEVICE: ${DEVICE}"
echo "MODEL_DTYPE: ${RESOLVED_MODEL_DTYPE}"
echo "ROUTE_MASS_RETENTION: ${ROUTE_MASS_RETENTION}"
echo "VALUE_ENERGY_RETENTION: ${VALUE_ENERGY_RETENTION}"
echo "VERIFY_HASHES: ${VERIFY_HASHES}"

args=(
  --split-root "${SPLIT_ROOT}"
  --source-json "${RAGTRUTH_RESPONSE_JSON}"
  --model-path "${MODEL_PATH}"
  --output-root "${OUT}"
  --device "${DEVICE}"
  --model-dtype "${RESOLVED_MODEL_DTYPE}"
  --route-mass-retention "${ROUTE_MASS_RETENTION}"
  --value-energy-retention "${VALUE_ENERGY_RETENTION}"
  --minimum-role-edges "${MINIMUM_ROLE_EDGES}"
  --conservation-atol "${CONSERVATION_ATOL}"
  --conservation-rtol "${CONSERVATION_RTOL}"
  --cache-binding-atol "${CACHE_BINDING_ATOL}"
  --output-dtype "${OUTPUT_DTYPE}"
)

[ -n "${LIMIT}" ] && args+=(--limit "${LIMIT}")
[ "${VERIFY_HASHES}" = "1" ] && args+=(--verify-hashes)
[ "${OVERWRITE}" = "1" ] && args+=(--overwrite)
[ "${ALLOW_REMOTE_FILES}" = "1" ] && args+=(--allow-remote-files)
[ "${TRUST_REMOTE_CODE}" = "1" ] && args+=(--trust-remote-code)
[ -n "${REVISION}" ] && args+=(--revision "${REVISION}")

run_stage "exact frozen operator graph construction" \
  "${PYTHON}" -m experiments.frozen_operator_graph.run "${args[@]}"

echo
echo "Finished successfully."
echo "Graph split: ${OUT}"
echo "Manifest: ${OUT}/manifest.json"
echo "Index: ${OUT}/index.jsonl"
