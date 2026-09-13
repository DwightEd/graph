# Native-v1 experiment integrity audit

Date: 2026-09-13  
Auditor: real collaboration GPT-5.5 fallback, xhigh-style integrity review. Codex MCP was unavailable, so this is not an external Codex MCP verdict and not an experiment-integrity auto-acquittal.  
Scope: `outputs/native_audit_v1_20260913/evaluation.json`, `settings.json`, `A/B/C/D/merge/*.json`, frozen `executed_code`, `outputs/native_audit_design_20260913/evaluation_manifest.json`, `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_TRACKER.md`, `docs/NATIVE_METHOD_MODEL_20260913.md`, and `refine-logs/native_batch_witness_20260913.md`.

## Overall verdict

Integrity status: **WARN**  
Scientific claim support: **NO**

I found no evidence that RAGTruth labels were used to create the method predictions, no fake ground truth derived from reader output, and no self-normalization fraud in the visible frozen evaluator. The run artifacts are real and internally hash-consistent. The scientific result is still a negative / non-supporting result: v1 produced no risk claims, no selected native windows, no B/C/D native payloads, zero native forward calls, zero mechanism-scored words, and no resolved mechanisms.

The main result that v1 supports is: the full frozen pipeline can execute through A/B/C/D/merge on 36 exploratory train responses while conservatively abstaining. It does **not** support claims of hallucination detection, automatic lookback, route-derived spans, native mechanism coverage, or graph-over-reader gain.

## Key measured facts

| Item | Value |
|---|---:|
| Responses | 36 |
| Sources | 6 |
| Tasks | 12 QA / 12 Summary / 12 Data2txt |
| Total words | 4,733 |
| RAGTruth annotated error words | 260 |
| A-stage questions | 172 |
| A-stage unsupported questions | 0 |
| A-stage native-selected questions | 0 |
| B/C/D payloads | 36 files each, all empty dicts |
| Native forward calls | 0 |
| Native tokens processed | 0 |
| Mechanism-scored words | 0 |
| Resolved mechanisms | 0 |

Evidence: `evaluation.json` reports 36 responses and 6 sources at lines 12-15, semantic coverage and mechanism coverage at lines 16-80, and the aggregate failure/mechanism counters at lines 301-313. A sample A artifact has `native_selected_question_ids: []` at `A/13801.json:1711`; a sample B/C/D chain has empty `data` at `B/13800.json:8`, `C/13800.json:9`, and `D/13800.json:10`; a sample merged output has empty graph nodes/groups and zero native calls at `merge/13800.json:1195-1218`.

## A. Ground truth provenance: PASS

The word-level evaluation labels come from the RAGTruth annotation file recorded in the pre-frozen manifest, not from Qwen reader outputs. The manifest points to `/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl` and records its SHA256 at `outputs/native_audit_design_20260913/evaluation_manifest.json:5-6`. It also records `annotation_read: false` at line 9 and a roster coverage contract at line 10.

The run settings bind the frozen 36-row input roster and the evaluation manifest: `settings.json:68-69` gives the input path/hash, and `settings.json:97-100` records the manifest path/hash. The protocol explicitly records `labels_used: false` and `reader_predictions_not_ground_truth: true` at `settings.json:64-66`.

The evaluator refuses to join labels until method predictions are complete, then rehashes the manifest, checks the label path, rehashes the input roster, and verifies each annotation's source ID, generator, split, and response hash before computing word labels. Evidence: `executed_code/experiments/evaluate_native_audit.py:64-83`, `:84-96`, and `:116-125`. The actual result records `labels_used_stage: evaluation_only_after_complete` at `evaluation.json:7`.

Classification: the word-level detection evaluation is **real_gt** for RAGTruth span annotations. The native mechanism side is **self_supervised_proxy / no independent mechanism GT**; the result file correctly says unique lookback accuracy is not identifiable without independent node ground truth at `evaluation.json:10`.

## B. Score normalization: WARN

No self-normalization fraud was found in the visible evaluator. The evaluator concatenates labels, scores, abstention flags, and source IDs, applies the frozen threshold, and computes coverage/precision/recall directly from those arrays at `executed_code/experiments/evaluate_native_audit.py:18-61`. It computes scores from merged word fields at `:143-152`. The semantic reader label probabilities are explicitly normalized only over enumerated labels, not over the model's own max output statistic, at `executed_code/route_graph/frozen_reader.py:108-123`.

The warning is reproducibility-related: `evaluate_native_audit.py` imports `binary_detection_metrics` from `route_graph.metrics` at line 15, but `metrics.py` is not present under `outputs/native_audit_v1_20260913/executed_code/route_graph/`, and `settings.json:72-89` does not include it in `code_sha256`. I inspected the current live `route_graph/metrics.py` only to understand likely behavior, but it is not part of the v1 frozen code snapshot. Therefore the audit can say no self-normalization is visible in the frozen evaluator body, but it cannot fully replay the AUROC/AUPRC implementation from the frozen artifact set alone.

This does not create a positive result: all mechanism ranking metrics are degenerate because mechanism scores are all abstentions/defaults; aggregate mechanism AUROC is 0.5 with zero valid bootstrap replicates at `evaluation.json:60-80`.

## C. Result file existence and number matching: PASS with reproducibility warning

All expected run result files exist for the frozen 36 responses:

- `A/*.json`: 36
- `B/*.json`: 36
- `C/*.json`: 36
- `D/*.json`: 36
- `merge/*.json`: 36

The run progress file reports `status: complete`, `responses: 36`, and `claim_supported: not_evaluated` at `progress.json:1-6`. The evaluator also records the output scope as six train sources and exploratory observer replay at `evaluation.json:8`, and `scientific_review: not_performed` at `evaluation.json:3134`.

I recomputed wrapper `content_sha256` and upstream file SHA256 links for all A/B/C/D/merge artifacts; no mismatch was found. I also checked that every `settings.json` code hash listed for frozen `executed_code/route_graph/*.py` matches the copied file, and the evaluator hash in `evaluation.json:6` matches `executed_code/experiments/evaluate_native_audit.py`.

The reproducibility warning is the missing frozen `route_graph/metrics.py` dependency described above. That is not a phantom result, but a sealed-replay gap for AUROC/AUPRC code provenance.

The documented wrapper witness is also not a clean end-to-end success witness: it states the documented invocation failed its complete witness gate with wrapper exit code 1, although the audit child exited 0 after all 36 responses passed A/B/C/D/merge. Evidence: `native_batch_witness_20260913.md:3`, `:21-31`, and `:47-53`. This does not invalidate the native output files, but it prevents claiming the wrapper/recovery protocol itself succeeded end-to-end.

## D. Dead code / called metrics: PASS with method-coverage warning

The evaluation metric function is actually called. `metrics()` is defined at `executed_code/experiments/evaluate_native_audit.py:18-61`, called for per-response metrics at `:189-194`, and called for task/all groups at `:205-210`.

The native search/validation machinery is not dead code in the implementation, but it was not exercised by this v1 result because A selected no risk questions. The selection logic chooses unsupported risk questions and optional recovery controls at `executed_code/route_graph/audit_prepare.py:53-85`, and later stages iterate only `selected_questions(anchors)` at `executed_code/route_graph/audit_runner.py:204-206` and `:254-302`. With `native_selected_question_ids` empty in every A artifact, B/C/D correctly publish empty data payloads.

Do not report phase completion as actual native intervention. In this run, stage files exist, but there were zero native forwards and no position/origin/paired/MLP/layer certificates.

## E. Scope assessment: FAIL for efficacy claims

The run has only six train sources and 36 responses, which is explicitly exploratory. Evidence: `evaluation.json:8`, `evaluation.json:12-15`, `docs/NATIVE_METHOD_MODEL_20260913.md:79-89`, and `FINAL_PROPOSAL.md:239-246`.

Even within that small scope, the effective method coverage is too low for positive claims:

- Semantic A/C scored only 287 / 4,733 words = 6.06%, with annotated error coverage 3 / 260 = 1.15%. Evidence: `evaluation.json:16-36` and `:38-58`.
- At the frozen 0.8 threshold, detected words are 0 and recall including abstentions is 0 for semantic A/C. Evidence: `evaluation.json:32-36` and `:54-58`.
- Mechanism scored words are 0 / 4,733, mechanism coverage is 0, and detected mechanism words are 0. Evidence: `evaluation.json:60-80`.
- Aggregate question status is 103 invalid, 56 uncertain, 13 supported, and 0 unsupported. Evidence: `evaluation.json:301-305`.
- Native status is only `not_selected_or_contrast_unavailable: 148`; no mechanisms, position statuses, or origin statuses are present. Evidence: `evaluation.json:307-313`.

The largest empirical bottleneck is A-stage semantic anchoring and contrast creation, not native graph search. An independent pass over A artifacts found 172 questions total, 103 invalid, 56 uncertain, 13 supported, 0 unsupported, and 0 selected native questions. Among invalid questions, 79 failed the self-QA recovery condition without a stored reason, 17 failed due to premise overlap with the hidden answer, and 7 failed due to absent/ambiguous quotes. A concrete self-QA overlong answer example is visible in `A/13801.json:67-93`; a premise-overlap invalid example is visible in `A/13800.json:64-80`.

## F. Evaluation type classification

- Word-level detection metric: **real_gt**, using RAGTruth span annotations after predictions complete.
- Semantic A/C reader output: **model-estimated proxy**, not GT. The method documents this at `FINAL_PROPOSAL.md:21-24`, `:69-77`, and `docs/NATIVE_METHOD_MODEL_20260913.md:24-30`.
- Native mechanism witnesses: **self_supervised_proxy / intervention audit**, with no independent mechanism ground truth. The method documents that no unique node truth exists at `docs/NATIVE_METHOD_MODEL_20260913.md:42-43`, and the result file records `unique_lookback_accuracy: not_identifiable_without_independent_node_ground_truth` at `evaluation.json:10`.
- Original generator internals: **not evaluated**. The run is observer replay with Llama 3.1, not the six generators' original traces; this is recorded in `settings.json:65`, `evaluation.json:8`, and `docs/NATIVE_METHOD_MODEL_20260913.md:87-89`.

## Claim impact

| Claim | Support |
|---|---|
| v1 completed frozen A/B/C/D/merge files for 36 exploratory responses | Supported, with wrapper witness caveat |
| RAGTruth labels were only joined after prediction completion | Supported |
| Reader predictions are not treated as GT | Supported in code/docs |
| Semantic A/C provides useful hallucination detection on this batch | Not supported |
| Native graph adds mechanism evidence over the semantic layer | Not supported |
| Automatic lookback / route-derived hallucination spans are demonstrated | Not supported |
| Unique internal node localization is measured | Not supported; explicitly not identifiable |
| One-4090 actual native budget is empirically characterized | Not supported by this run, because zero native forwards occurred |

## Action items

1. Treat native-v1 as a negative coverage result, not as a positive method result.
2. Do not report `AUROC=0.525` without the accompanying 6.06% semantic coverage, 1.15% annotated-error coverage, 0 threshold detections, and zero native forwards.
3. Do not use reader outputs as ground truth; keep RAGTruth labels as evaluation-only and mechanism outputs as no-GT intervention witnesses.
4. Freeze or vendor `route_graph/metrics.py` into `executed_code` and `settings.json` for any future sealed evaluation artifact.
5. Make the next iteration attack the A-stage bottleneck: shorter atomic role questions, non-overlap premise construction, and self-QA answer-span equality. V2 must still preserve the old double-blind source QA and certificate gates.

