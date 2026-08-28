#!/usr/bin/env bash

fail_run() {
  local message=$1
  local code=${2:-1}
  echo >&2
  echo "ERROR: ${message}" >&2
  echo "Stopped immediately; no later stage was run." >&2
  exit "${code}"
}

run_stage() {
  local stage_name=$1
  local stage_status
  shift

  # Deliberately do not redirect or capture the command output.  In particular,
  # Python tracebacks and KeyboardInterrupt remain visible in the terminal.
  "$@"
  stage_status=$?
  if [ "${stage_status}" -ne 0 ]; then
    fail_run "${stage_name} failed with exit code ${stage_status}." "${stage_status}"
  fi
}

REPO=${REPO:-/share/home/tm902089733300000/a903202310/lys/research/graph}
TEST_SPLIT=${TEST_SPLIT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test}
RAGTRUTH_ROOT=${RAGTRUTH_ROOT:-/share/home/tm902089733300000/a903202310/lys/data/RAGTruth}
SOURCE_INFO=${SOURCE_INFO:-${RAGTRUTH_ROOT}/source_info.jsonl}
if [ ! -f "${SOURCE_INFO}" ] && [ -f "${RAGTRUTH_ROOT}/dataset/source_info.jsonl" ]; then
  SOURCE_INFO=${RAGTRUTH_ROOT}/dataset/source_info.jsonl
fi
MODEL_PATH=${MODEL_PATH:-/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
TORCH_DTYPE=${TORCH_DTYPE:-auto}
TASK=${TASK:-QA}
SEED=${SEED:-20260828}
VOCAB_CHUNK_SIZE=${VOCAB_CHUNK_SIZE:-4096}
GRADIENT_PROBES=${GRADIENT_PROBES:-8}
ROLE_NULL_BIN_WIDTH=${ROLE_NULL_BIN_WIDTH:-32}
LIMIT=${LIMIT:-}
BOOTSTRAP=${BOOTSTRAP:-1000}
FOLDS=${FOLDS:-5}
START_STAGE=${START_STAGE:-1}
FORCE_ROLES=${FORCE_ROLES:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-0}

if ! [[ "${START_STAGE}" =~ ^[1-3]$ ]]; then
  fail_run "START_STAGE must be 1, 2, or 3."
fi

MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}")}
OUT=${OUT:-${REPO}/experiments/attention_mechanism_audit/outputs/${TASK,,}/${MODEL_TAG}_seed${SEED}}
ROLE_INDEX=${ROLE_INDEX:-${OUT}/prompt_roles.jsonl}
ARTIFACT=${ARTIFACT:-${OUT}/mechanisms.npz}
EVALUATION=${EVALUATION:-${OUT}/evaluation.json}

run_stage "enter repository ${REPO}" cd "${REPO}"
run_stage "create output directory ${OUT}" mkdir -p "${OUT}"

echo "EXPERIMENT: attention_mechanism_audit"
echo "OUTPUT: ${OUT}"

LIMIT_ARGUMENT=()
[ -n "${LIMIT}" ] && LIMIT_ARGUMENT=(--limit "${LIMIT}")
TRUST_ARGUMENT=()
[ "${TRUST_REMOTE_CODE}" = "1" ] && TRUST_ARGUMENT=(--trust-remote-code)

if [ "${START_STAGE}" -le 1 ]; then
  if [ ! -f "${SOURCE_INFO}" ]; then
    fail_run "SOURCE_INFO must point to RAGTruth's label-free source_info.jsonl: ${SOURCE_INFO}"
  fi
  if [ -f "${ROLE_INDEX}" ] && [ "${FORCE_ROLES}" != "1" ]; then
    echo
    echo "[1/3] Reuse exact prompt-role index: ${ROLE_INDEX}"
  else
    echo
    echo "[1/3] Reconstruct and verify label-free prompt roles"
    run_stage "prompt-role reconstruction" \
      "${PYTHON}" -m experiments.attention_mechanism_audit.run roles \
      --data "${TEST_SPLIT}" \
      --source-info "${SOURCE_INFO}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --output "${ROLE_INDEX}" \
      --task "${TASK}" \
      "${LIMIT_ARGUMENT[@]}" \
      "${TRUST_ARGUMENT[@]}"
  fi
else
  echo
  echo "[1/3] Prompt-role reconstruction -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${ROLE_INDEX}" ]; then
  fail_run "Prompt-role index is missing: ${ROLE_INDEX}"
fi

if [ "${START_STAGE}" -le 2 ]; then
  if [ ! -f "${SOURCE_INFO}" ]; then
    fail_run "SOURCE_INFO is required to bind the frozen mechanism artifact."
  fi
  echo
  echo "[2/3] Replay the frozen model and capture three separate mechanisms"
  run_stage "mechanism capture" \
    "${PYTHON}" -m experiments.attention_mechanism_audit.run capture \
    --data "${TEST_SPLIT}" \
    --roles "${ROLE_INDEX}" \
    --source-info "${SOURCE_INFO}" \
    --model "${MODEL_PATH}" \
    --output "${ARTIFACT}" \
    --device "${DEVICE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --task "${TASK}" \
    --vocab-chunk-size "${VOCAB_CHUNK_SIZE}" \
    --gradient-probes "${GRADIENT_PROBES}" \
    --attribution-seed "${SEED}" \
    --role-null-bin-width "${ROLE_NULL_BIN_WIDTH}" \
    "${LIMIT_ARGUMENT[@]}" \
    "${TRUST_ARGUMENT[@]}"
else
  echo
  echo "[2/3] Mechanism capture -- skipped (START_STAGE=${START_STAGE})"
fi

if [ ! -f "${ARTIFACT}" ]; then
  fail_run "Mechanism artifact is missing: ${ARTIFACT}"
fi

echo
echo "[3/3] Freeze artifact bytes, then open labels for post-hoc evaluation"
run_stage "post-hoc mechanism evaluation" \
  "${PYTHON}" -m experiments.attention_mechanism_audit.run evaluate \
  --data "${TEST_SPLIT}" \
  --artifact "${ARTIFACT}" \
  --output "${EVALUATION}" \
  --bootstrap "${BOOTSTRAP}" \
  --folds "${FOLDS}" \
  --seed "${SEED}"

render_mechanism_report() {
  "${PYTHON}" - "${EVALUATION}" <<'PY'
import json
import sys


path = sys.argv[1]
with open(path, encoding="utf-8") as file:
    report = json.load(file)

expected_schema = "attention-hallucination-mechanism-answer-evaluation"
schema = report.get("schema")
if schema != expected_schema:
    raise ValueError(
        f"wrong evaluation schema: expected {expected_schema!r}, got {schema!r}; "
        "refusing to print an old operator-validation report"
    )

expected_features = (
    "drift_functional_history_to_grounding_log_ratio"
    "__layer_mean__late_minus_early",
    "dispersion_functional_entropy_observed__layer_mean__late_minus_early",
    "dispersion_functional_cancellation__layer_mean__late_minus_early",
    "routing_entropy_upper__layer_mean__late_minus_early",
    "routing_total_evidence_ancestry__layer_mean__late_minus_early",
    "counterfactual_evidence_bypass__mean",
)
declared_features = tuple(report.get("primary_answer_feature_names", ()))
if declared_features != expected_features:
    raise ValueError(
        "mechanism primary endpoints differ from the frozen six-endpoint audit: "
        f"{declared_features!r}"
    )


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


rows = {
    row.get("feature"): row
    for row in report.get("primary_answer_univariate", ())
}
missing_rows = [name for name in expected_features if name not in rows]
if missing_rows:
    raise ValueError(f"primary mechanism rows are missing: {missing_rows}")

print("\n=== ATTENTION HALLUCINATION MECHANISM AUDIT ===")
print(f"schema: {schema}")
print(f"samples: {report.get('samples')}")
print(f"positive_answers: {report.get('positive_answers')}")
print(f"prevalence: {fmt(report.get('prevalence'))}")

print("\n=== SIX FROZEN PRIMARY MECHANISM TESTS ===")
for name in expected_features:
    row = rows[name]
    direction = row.get("direction")
    metrics = row.get("oriented") or {}
    permutation = row.get("source_group_permutation") or {}
    raw_effect = permutation.get("mean_positive_minus_negative")
    if raw_effect is not None and direction == "low":
        oriented_effect = -raw_effect
    else:
        oriented_effect = raw_effect
    print(name)
    print(
        "  "
        f"direction={direction} "
        f"AUROC={fmt(metrics.get('auroc'))} "
        f"AUPRC={fmt(metrics.get('auprc'))} "
        f"oriented_source_effect={fmt(oriented_effect)} "
        f"p={fmt(permutation.get('p_value_two_sided'))} "
        f"BH_FDR_q={fmt(row.get('source_group_permutation_fdr_q'))}"
    )

print("\n=== INCREMENT OVER PROMPT + RESPONSE LENGTH ===")
increments = report.get("primary_feature_length_increment", {})
for name in expected_features:
    result = increments.get(name, {})
    print(
        f"{name}\n"
        "  "
        f"available={result.get('available')} "
        f"delta_AUROC={fmt(result.get('auroc_delta'))} "
        f"delta_AUPRC={fmt(result.get('auprc_delta'))}"
    )

print("\n=== TOKEN FIRST-ONSET DIAGNOSTICS ===")
onset = report.get("token_onset_diagnostics", {})
if not onset:
    print("unavailable")
for name, result in onset.items():
    bootstrap = result.get("source_bootstrap") or {}
    print(name)
    print(
        "  "
        f"answers={result.get('responses_with_first_onset')} "
        f"matched={result.get('source_disjoint_same_position_matches')} "
        "onset_minus_control="
        f"{fmt(result.get('mean_onset_minus_matched_non_onset_delta'))} "
        f"95%CI=[{fmt(bootstrap.get('ci_low'))}, "
        f"{fmt(bootstrap.get('ci_high'))}]"
    )

print("\n=== MECHANISM OBSERVABILITY ===")
observability = report.get("mechanism_observability", {})
if not observability:
    print("unavailable")
for name in sorted(observability):
    print(f"{name}: {observability[name]}")

print("\n=== CLAIM BOUNDARY ===")
print(report.get("claim_boundary", "unavailable"))
print(f"\nFull report: {path}")
PY
}

# This call is reached only after evaluation returned exit code zero.  A failed
# or interrupted capture/evaluation can therefore never fall through to JSON.
run_stage "mechanism report rendering" render_mechanism_report

echo
echo "Finished: ${OUT}"
echo "Mechanism artifact: ${ARTIFACT}"
echo "Evaluation: ${EVALUATION}"
