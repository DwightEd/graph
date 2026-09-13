# Native v3: verbatim cloze anchors, full 36-response development audit

Existing research environment; no package installation or rebuild. The unchanged six-source roster is development data after v1/v2 evaluations. No new generalization or effectiveness claim. V1 had 6.06% semantic coverage, v2 had zero; both executed zero native forwards. V3 changes the failed semantic interface, not the causal thresholds, candidate controls, annotation policy or native budgets.

## Inputs, preparation and scope

Work from `/share/home/tm902089733300000/a903202310/lys/research/graph`.
Input `outputs/native_audit_design_20260913/inputs.jsonl` is 36 responses, two train sources per task with all six generators. SHA256 `c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264`. Labels never enter either model; exact original annotation manifest is bound before predictions. All 36 original observer tokenizations passed, maximum sequence length 1757.
Output `outputs/native_audit_v3_20260913`; do not overwrite v1/v2 or use their artifacts as v3.

Author CPU preparation, already separately requested (the fresh witness does not repeat):

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m route_graph.audit_runner --inputs outputs/native_audit_design_20260913/inputs.jsonl --output outputs/native_audit_v3_20260913 --evaluation-manifest outputs/native_audit_design_20260913/evaluation_manifest.json --stage prepare

This freezes source hashes, model/tokenizer bytes, protocol, runtime and input hashes; no GPU model load. Prepared artifacts are verified on resume, never silently rewritten. The author saves matching executed source copies before the full run.

## Fresh witness: execute exactly this invocation once

Working directory `/share/home/tm902089733300000/a903202310/lys/research/graph`:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_native_audit.py --population-pid 145947 --output outputs/native_audit_v3_20260913 --journal ../reanchor/runs/native_audit_v3_20260913.json

Use a persistent exec session; preserve it while waiting. Do not terminate the scheduler. Report its session ID and later the exit code. Do not improvise a different invocation, fix code, retry or launch another GPU task. A failure is a documented divergence for the parent to diagnose.

The scheduler verifies prepared native artifacts and the original population's frozen settings/code before any signal, pauses population gracefully using a Linux PID descriptor, and retains exclusive run locks. It executes A (Qwen cloze anchors), B (Llama candidate search), C (Qwen blind fixed-pool relation estimates), D (Llama fixed controls), then CPU merge for every response. Models load sequentially and are unloaded between phases. It preserves all previously completed population manifest hashes and restores the original population in its finally path. Restoration has a 900-second deadline and requires actual matching PID/progress state and held output lock; a launcher or preparing state is insufficient. Do not report witness success until both native execution and verified restoration succeed.

## What differs in v3

Atomic frame extraction remains bounded: four response units per batch, at most eight calls and 48 frame proposals per response, including rejected proposals; at most 48 compiled questions. Clauses keep original non-target text; target role alone becomes `<MISSING_SLOT>`. Non-target pronouns use validated earlier referents. Unassigned alphanumeric text remains named required conditions, not a discarded qualifier.
The self check receives the current assertion solely to test mask recoverability. Both source passes receive only source, masked question, non-target condition table and task. Neither gets the unmasked assertion, response answer, aliases or previous source answer. Every condition has an exact role-specific citation binding. Equal referent text in two roles keeps distinct coordinates, checked with canonical sidecar digest, question hash and original response spans.
Reader raw output is preserved byte-for-byte after decoding. Only one intact JSON object is accepted; explicitly recorded fences or trailing closing punctuation may be recovered. Incomplete content, duplicate keys, second roots and prose tails fail. Generation/context-budget failures are failures even if a complete root was present. Every returned request, including cache hits, is counted in A/C and merge artifacts with strict/recovered/failure outcomes. Boundary parsing does not make semantic predictions correct.
Native budgets remain at most four distinct claims/response, 320 actual forwards/claim and 1280/response; an eligible correct-recovery control reserves one claim. All words, abstentions, skipped groups and failed controls stay in the output. Completed phase files do not prove any native intervention took place.

## Read-only monitoring

Progress `outputs/native_audit_v3_20260913/progress.json`; audit log `../reanchor/runs/native_audit_v3_20260913.audit.log`; journal `../reanchor/runs/native_audit_v3_20260913.json`; restored population log `../reanchor/runs/native_audit_v3_20260913.population.log`; population progress `../reanchor/outputs/ragtruth_population_20260912/progress.json`.
Optional CPU summary: `python -m experiments.summarize_native_audit --run outputs/native_audit_v3_20260913` from graph, using the documented research Python. It does not join labels.

## Independent evaluation after completion (parent only)

After native progress is complete and all immutable predictions are published:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.evaluate_native_audit --run outputs/native_audit_v3_20260913 --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl --output outputs/native_audit_v3_20260913/evaluation.json

Use the module invocation. Compare semantic-only versus mechanism-covered outputs with abstentions in all denominators. Record native forwards, selected windows, control failures and raw-origin versus position certificates. The Llama observer replays six models' response texts; it is not their original internal trace. There is no independently annotated unique lookback node or causal endpoint in RAGTruth; never report 100% lookback accuracy or infer absence of a dependency from failure to measure it. A second JSON parser or a successful interface alone does not resolve routing, ownership or continuous hallucination spans.
