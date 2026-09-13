# Native v2: atomic event-role anchor, complete 36-response development batch

This invocation uses the existing research environment, without installs or a rebuild.
It runs all 36 preselected natural responses through A (Qwen question/constraint estimates),
B (Llama native candidate search), C (Qwen blinded candidate and slot-relation estimates),
D (Llama fixed causal controls), then CPU merge. This is an exploratory train-source
audit; it does not establish test-set accuracy or original-generator causal traces.

## Prepared inputs and outputs

Input: outputs/native_audit_design_20260913/inputs.jsonl.
SHA256: c2d5712bf5f08ddaf34427eb483e98f38433d9a09847a857efe44bbf0de0b264.
Six source IDs, two per task, all six generators per source; 36 responses.
All exact tokenizations passed; maximum observed sequence length 1757.
Output: outputs/native_audit_v2_20260913. Do not overwrite old run directories.
Prepare freezes protocol, every code file, model/tokenizer bytes, and input hash.
No labels are passed to either model. Reader predictions remain fallible estimates.
Up to four distinct claims/response, including one eligible supported recovery control.
Up to 320 native forwards/claim and 1280/response; all failures and abstentions remain.

## CPU preparation (author executes after engineering review)

From the graph repository:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m route_graph.audit_runner --inputs outputs/native_audit_design_20260913/inputs.jsonl --output outputs/native_audit_v2_20260913 --evaluation-manifest outputs/native_audit_design_20260913/evaluation_manifest.json --stage prepare

This imports packages and hashes local files, but does not load either model on GPU.
Existing prepared outputs are verified, never silently rewritten.

## Full natural batch (fresh doc witness executes once)

Working directory:

    /share/home/tm902089733300000/a903202310/lys/research/graph

Execute this exact invocation once, after the preparation and review have passed:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u experiments/interleave_native_audit.py --population-pid 141861 --output outputs/native_audit_v2_20260913 --journal ../reanchor/runs/native_audit_v2_20260913.json

Use a persistent exec session; do not terminate the scheduler while waiting.
The scheduler verifies the original frozen population and prepared audit before any signal.
It uses a Linux PID descriptor, gracefully pauses the existing population,
holds both output locks, and gives its audit child the already-owned audit lock.
All phases execute sequentially; each model is unloaded before the next load.
It verifies all completed population manifests remain unchanged before restoring the
original population. Restoration is checked through actual progress PID and output lock.
If the wrapper exits or refuses preflight, report the log/journal and do not improvise
another launch or retry. Parent handles diagnosis with a fresh output when required.

## Read-only monitoring

Audit progress: outputs/native_audit_v2_20260913/progress.json.
Audit log: ../reanchor/runs/native_audit_v2_20260913.audit.log.
Scheduler journal: ../reanchor/runs/native_audit_v2_20260913.json.
Restored population log: ../reanchor/runs/native_audit_v2_20260913.population.log.
Population progress: ../reanchor/outputs/ragtruth_population_20260912/progress.json.

Do not report witness success until all phases exit successfully AND population resume
is verified. An A-stage question or completed JSON file is not a successful detection.
The annotation join happens only after immutable method predictions are published.

## Expected scientific outputs and boundaries

Record semantic risk coverage, exact counterfactual availability, native candidate
coverage, position versus raw-input-origin certificates, contrast-template stability,
and matched unrelated-control failures. Compare semantic-only and mechanism-covered
predictions with abstentions included, rather than only reporting passing examples.
Graph nodes preserve assertion/answer spans and native query/key/layer groups.
An error-continuation edge additionally requires a frozen prior-question mapping and
a positive donor effect originating at that prior answer slot, with both slots estimated
unsupported. Correct recovery remains a separate control. An unlinked history effect
cannot establish continuous error. No unique or 100%-accurate lookback claim is made.

The pre-run evaluation manifest records the original frozen population dataset digest. Native settings bind its bytes without reading annotations. All full-dataset annotation IDs remain outside the model; the evaluator joins only the exact 36 roster identities after completion.


## Version 2 changes and actual prior outcome

V1 child completed all36 predictions but had0 selected windows/0 native forwards. Its wrapper exited1 during the120-second restoration deadline; subsequent independent evidence confirms populationPID141861 resumed and advanced. That original failure journal is preserved. V2 extends only the restoration-verification timeout to900 seconds; it still requires matching PID/progress and actual held output lock. A launcher or preparing status alone is not success.

V2 uses atomic event-role extraction and exact span/antecedent bindings. Deterministic questions hide one role and retain every other role. Both blind source QA passes must return an exhaustive role-specific condition table. Target absence cannot be inferred from an unresolved non-target premise. Budgets: max8 frame batches/response,4 units/batch,24 frames/batch and48 frame proposals/response including invalid proposals,48 final role questions. All context remains available; omitted units remain explicitly unknown. Source JSON limit1536; frame JSON limit3072. Native thresholds/budgets/control protocol otherwise unchanged. Engineering review closed;25 relatedCPU tests passed. No factual-effectiveness claim.

This is the same six-source development roster used in v1. No label-based threshold change or favorable-row selection. A later test claim requires new untouched sources. A–D exit0 must not be equated with actual interventions; count native forwards and selected windows from outputs.

## Independent evaluation after all predictions complete

From graph, execute only after progress status complete (module invocation is required):

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.evaluate_native_audit --run outputs/native_audit_v2_20260913 --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl --output outputs/native_audit_v2_20260913/evaluation.json

The fresh witness executes only the full natural batch invocation above, not this evaluation. Parent separately evaluates frozen predictions. Preserve all invalid/unknown examples. No hidden repair or retry.
