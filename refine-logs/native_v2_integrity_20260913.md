# Native audit v2 integrity and result-to-claim audit

Date: 2026-09-13
Auditor: real collaboration GPT-5.5 fallback. Codex MCP was unavailable and was not impersonated. This is an independent file-based audit of the frozen v2 run snapshot, not an external MCP verdict.

Scope read: `outputs/native_audit_v2_20260913/evaluation.json`, `settings.json`, `progress.json`, bounded A/B/C/D/merge artifacts, `outputs/native_audit_v2_20260913/executed_code`, `outputs/native_audit_design_20260913/evaluation_manifest.json`, `refine-logs/FINAL_PROPOSAL.md`, `docs/NATIVE_METHOD_MODEL_20260913.md`, and the existing tracker/witness files. I did not read or judge current v3 code, did not modify run outputs or implementation, and did not run GPU.

## Overall verdict

Integrity verdict: **WARN, not FAIL**. The files support a real completed v2 run with frozen inputs/code, independent RAGTruth labels joined only during evaluation, and no score-normalization fraud. The warning is scientific scope: the run produced **zero scored semantic words, zero selected native questions, zero native forward calls, and zero mechanism certificates**. It is therefore a valid negative/diagnostic run, not evidence that the method works.

Overall `claim_supported`: **no** for the scientific method claims. The only supported claim is the execution/integrity claim that the 36-response exploratory batch completed and was evaluated after freezing predictions.

## Mechanical evidence pre-check

- Run progress says `status=complete`, `responses=36`, `claim_supported=not_evaluated`: `outputs/native_audit_v2_20260913/progress.json:1-6`.
- Stage file counts are present for all phases: 36 A, 36 B, 36 C, 36 D, 36 merge artifacts. I recomputed every stage envelope's `content_sha256` and upstream checksums; no mismatch was found.
- `settings.json` freezes protocol, input path, model paths, code hashes, seed, runtime, and evaluation manifest: `outputs/native_audit_v2_20260913/settings.json:1-112`.
- The code hashes in `settings.json` match the `executed_code/route_graph/*.py` snapshot for every listed file. I found no executed-code mismatch.
- `native_v2_doc_witness_20260913.md` and `reanchor/runs/native_audit_v2_20260913.json` were not present at audit time. Therefore there is no separate v2 wrapper/doc witness to cite. The run artifacts themselves are internally consistent, but no independent wrapper success should be claimed.

## A. Ground-truth provenance: PASS with scope caveat

The evaluation labels are dataset annotations, not reader/model outputs. The design manifest points to RAGTruth `response.jsonl`, stores its SHA256, and says annotations were not read when freezing the roster: `outputs/native_audit_design_20260913/evaluation_manifest.json:5-10`. The v2 settings record `labels_used=false`, `reader_predictions_not_ground_truth=true`, and `observer_replay_not_original_generator_internal_trace=true`: `outputs/native_audit_v2_20260913/settings.json:73-76`.

The evaluator refuses to run unless all method predictions are complete, then verifies the pre-frozen manifest, input hash, label path, label file hash, source id, generator, split, and response hash before constructing word labels: `outputs/native_audit_v2_20260913/executed_code/experiments/evaluate_native_audit.py:64-125`. It constructs labels only from annotation spans after reading merge/A/D artifacts: `outputs/native_audit_v2_20260913/executed_code/experiments/evaluate_native_audit.py:126-152`.

Caveat: this is a 36-row exploratory train-source subset, not an official full RAGTruth test evaluation. The evaluation itself states `scope=six_train_sources_exploratory_observer_replay_not_original_generator_trace`, no confidence intervals from only six sources, and no unique lookback accuracy: `outputs/native_audit_v2_20260913/evaluation.json:7-10`.

## B. Score normalization: PASS with reporting caveat

I found no metric normalized by the model's own maximum, minimum, or mean prediction. The evaluation computes word-level AUROC/AUPRC with sklearn and optional source-balanced sample weights: `outputs/native_audit_v2_20260913/executed_code/route_graph/metrics.py:9-40`, `outputs/native_audit_v2_20260913/executed_code/route_graph/metrics.py:82-84`. The evaluation script reports raw coverage, detected words, precision, recall, and covered error rate without tuning thresholds from labels: `outputs/native_audit_v2_20260913/executed_code/experiments/evaluate_native_audit.py:18-61`.

The reader's E/C/N/U comparison uses a softmax conditional on enumerated labels and records that normalization: `outputs/native_audit_v2_20260913/executed_code/route_graph/frozen_reader.py:108-123`. That is an internal reader estimate, not the final detection metric or ground truth.

Reporting caveat: because every word abstained and default scores are 0.5, AUROC=0.5 and AUPRC is effectively the source-balanced prevalence diagnostic. For ALL, `scored_words=0`, `coverage=0.0`, AUROC=0.5, AUPRC=0.05456, detected words=0, recall=0: `outputs/native_audit_v2_20260913/evaluation.json:16-80`. These numbers must not be described as meaningful ranking performance.

## C. Result file existence and number consistency: PASS

The run has the required 36 files in each of A/B/C/D/merge. The evaluator output records 36 responses and 6 sources overall: `outputs/native_audit_v2_20260913/evaluation.json:11-15`. It records the full failure counts: 93 invalid questions, 26 uncertain questions, 119 native statuses all `not_selected_or_contrast_unavailable`, and no mechanisms/position/origin statuses: `outputs/native_audit_v2_20260913/evaluation.json:301-312`.

My recomputation from per-response artifacts matched the evaluation summary:

- A questions: 119 total = 93 `invalid_question` + 26 `uncertain` + 0 `supported` + 0 `unsupported`.
- `native_selected_question_ids`: empty for all 36 responses.
- B/C/D artifact item totals: 0/0/0.
- Merge native forward calls and tokens processed: 0/0.
- Word states: 4053 `coverage_unknown`, 545 `assertion`, 135 `nonassertion` over 4733 words.

A representative A artifact shows 33 proposed questions but no selected native questions and zero risk claims: `outputs/native_audit_v2_20260913/A/13803.json:13130-13138`, `outputs/native_audit_v2_20260913/A/13803.json:16566-16569`. Its B and D artifacts are empty objects with valid upstream hashes: `outputs/native_audit_v2_20260913/B/13803.json:1-9`, `outputs/native_audit_v2_20260913/D/13803.json:1-10`. Its merge result records semantic and mechanism scored words as 0 and forward calls/tokens as 0: `outputs/native_audit_v2_20260913/merge/13803.json:6978-6999`.

## D. Dead code / actual-call detection: WARN

The native code exists in the executed snapshot and the phase runner would call it only for selected A questions. The runner selects `native_selected_question_ids` from A and then loops over those selected questions for B/C/D: `outputs/native_audit_v2_20260913/executed_code/route_graph/audit_runner.py:206-305`. A prepares selected questions only from unsupported risks and optional recoveries: `outputs/native_audit_v2_20260913/executed_code/route_graph/audit_prepare.py:55-88`. Since A produced no unsupported or supported-recovery control, B/C/D correctly had no selected questions.

This is not dead-code fraud if the reported claim is “v2 failed before native validation.” It becomes a severe claim error if anyone writes that v2 tested route groups, origin mediation, paired R, MLP interactions, layer bands, automatic lookback, or route-derived continuous spans. The run did not execute those causal measurements on any claim.

Actual calls:

- Reader requests executed: 274 cache files / A reader calls. Request types in the cache: 119 SELF, 86 FRAME, 26 SOURCE, 17 VERIFY, 26 COMPARE. Reader errors: 58 `invalid_json`; A frame calls include 51 `invalid_json`.
- Native causal oracle forward calls: 0, as recorded in every merge artifact and in the aggregate `forward_calls=0`, `native_tokens_processed=0` fields.

## E. Scope assessment: WARN for integrity, FAIL for broad scientific claims

The planned natural batch was 6 train sources, 36 responses, three tasks, at most 4 claims/response and 320 native forwards/claim; docs describe this as exploratory and not cross-dataset confirmation: `docs/NATIVE_METHOD_MODEL_20260913.md:85-95`, `refine-logs/FINAL_PROPOSAL.md:236-242`. The actual roster is 36 train rows: QA 12, Summary 12, Data2txt 12, with 6 source IDs and 6 generators. This is enough to expose the current A-stage bottleneck but not enough to support general method claims.

The frozen success gates in the proposal require ≥80% valid semantic assertion-word coverage, ≥70% risk-claim contrast construction, and ≥50% complete internal selective certificates: `refine-logs/FINAL_PROPOSAL.md:201-203`, `refine-logs/FINAL_PROPOSAL.md:238-240`. v2 achieved 0.0 semantic coverage, 0 risk claims with valid contrast, and 0 internal certificates. These gates are not marginally missed; they are completely unmet.

## F. Evaluation type classification

- Dataset label evaluation: **real_gt**, using RAGTruth annotation spans joined only after predictions froze.
- Semantic A labels: **frozen-reader proxy**, explicitly not ground truth.
- Native mechanism validation: **self-supervised/causal replay proxy**, but in this run no native validation instance executed.
- Unique lookback/location truth: **unavailable**; the evaluator explicitly marks it not identifiable without independent node ground truth.

## Four core claim judgments

1. **Execution/integrity claim: v2 completed the frozen 36-response exploratory batch with labels joined only after method outputs.**

   `claim_supported`: **yes, with witness caveat**. The artifacts, progress, settings, hashes, and evaluation code support this. Caveat: no separate v2 doc-witness/journal file was found, so do not claim independent wrapper success.

2. **Semantic-anchor claim: atomic role QA can produce usable supported/unsupported anchors on the natural batch.**

   `claim_supported`: **no**. v2 produced 119 questions, but 93 were invalid and 26 uncertain; there were 0 supported and 0 unsupported semantic-valid questions. ALL semantic coverage is 0/4733 words. Representative failure: the compiled target was `in regulating blood sugar levels`, but SELF_QA returned `a role` and the question was invalidated: `outputs/native_audit_v2_20260913/A/13803.json:3378-3394`.

3. **Native mechanism claim: candidate-blind native search and fixed D validation identify position/origin/pairedR/MLP/layer-band witnesses.**

   `claim_supported`: **no**. A selected zero questions in all 36 responses, so B/C/D artifacts contain no per-question entries and total native forward calls are 0. The native code was present, but this run did not measure a single finite event contrast.

4. **End-to-end method claim: adding native graph evidence improves hallucination detection, automatic lookback, condition attribution, or route-derived continuous spans.**

   `claim_supported`: **no**. There were 260 annotated error words, but scored words=0, detected words=0, recall including abstentions=0, mechanism scored words=0, and no conditional history edges or mechanisms. The evaluator states unique lookback accuracy is not identifiable and the graph flag `unique_lookback_node_claim` is false in merge artifacts.

## Claim impact and required next step

The v2 result supports one narrow diagnostic statement: **the frozen v2 atomic-role JSON interface failed at A on the 36-response exploratory natural batch, blocking all downstream native causal validation**. It does not support claims about hallucination detection, internal information insufficiency, route correctness, automatic lookback, condition attribution, continuous span boundaries, or mechanism class coverage.

The next method step should be judged as an A-interface repair, not as a native-mechanism expansion. The proposed cloze/first-root-parser v3 direction may be justified by the failure profile, but it must be evaluated as a new run and cannot borrow v2's completed B/C/D phase labels as evidence of mechanism execution.
