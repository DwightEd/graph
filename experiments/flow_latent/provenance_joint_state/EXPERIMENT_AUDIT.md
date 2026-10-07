# Experiment Audit Report

**Date:** 2026-10-08  
**Auditor:** requested GPT-5.6-Sol ultra fresh reviewer; runtime model attestation unavailable  
**Independence:** same-family, provisional  
**Project:** provenance_joint_state  
**Trace path:** `.aris/traces/experiment-audit/2026-10-08_provenance_joint_state/`

## Overall Verdict: WARN

## Integrity Status: warn

No fake ground truth, self-max score normalization, phantom result, or dead metric path was found. The saved numbers and run states are internally consistent and independently reproducible from the frozen artifacts.

The warning is an evidence-ceiling warning. The final evaluation contains only 17 previously exposed regression answers, the three seeds are optimization/control seeds on the same fixed cohort, and PJ005 blind confirmation is explicitly `NOT RUN`. The current documents respect this boundary and do not claim an effective detector or graph necessity.

## A. Ground Truth Provenance: PASS

The main hallucination metrics use dataset-provided RAGTruth character spans:

- `graph/experiments/token_backtrace/grounded_projection_evaluate.py:39-54` loads `response.jsonl`, requires a non-null dataset label, checks exact response and source identity, and aligns the dataset spans to persisted tokens.
- `graph/experiments/native_support/ragtruth.py:53-75` performs the character-span/token overlap and records `annotation_origin = RAGTruth/response.jsonl`.
- `graph/outputs/provenance_joint_state_20261008_v4_converged/evaluation_annotations.json:6686-7783` is one concrete saved example of the dataset-origin path; every one of the 17 saved annotations records the same origin.
- Independent reconstruction from `/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl` found all 17 regression IDs, exact response text, exact source IDs, and zero token-label mismatches. Relevant raw rows include `response.jsonl:12220` for ID 12219 and `response.jsonl:12298` for ID 12297. The full dataset parsed as 17,790 unique IDs with no duplicate IDs.

The fit target is not factual ground truth. `source_gap` is derived from native and source-blocked observer-model log probabilities (`measure.py:46-62`, `measure.py:205-208`), and rank selection predicts that anchor without natural labels (`run.py:55-91`). This is explicitly described as an unlabeled proxy/self-supervised target in `METHOD_IMPLEMENTATION.md:45-55`. It is not used as the evaluation truth.

The supervised logistic probe does use RAGTruth labels, but it is isolated as a diagnostic: `evaluate.py:62-108` declares that the weights do not enter the detector, and `supervised_diagnostic.json:2-7` records `diagnostic_only=true`, `natural_labels_used=true`, and `weights_transferred_to_detector=false`.

## B. Metric Normalization: PASS

No metric is divided by the model's own maximum, minimum, or mean to inflate performance.

- Task-specific calibration uses a source-disjoint reference cohort and empirical mid-ranks (`run.py:113-139`). The pre-calibration arrays are retained as `raw_*` scores (`run.py:132-138`).
- Raw-scale AUROC/AP is evaluated separately in `evaluate.py:149-156`; per-seed metrics are also recomputed from each seed's scores in `evaluate.py:157-174`.
- AUROC/AP are computed directly from dataset labels and scores (`grounded_projection_evaluate.py:57-79`). The independent tie-aware implementation in `audit_numeric.py:19-31` reproduced all stored values.
- The vector normalization in `grounded_projection.py:98-103` is a representation-norm control, not normalization of an evaluation metric.
- The documents correctly state that reference percentiles are a scale, not a hallucination probability or a guaranteed 5% false-positive rate (`RESULTS.md:35-37`; `METHOD_IMPLEMENTATION.md:53-55`).

The saved tables include raw metrics for the new methods. The historical `legacy_fixed` comparator is evaluated after reference-rank calibration and does not have a separate `raw_metrics` entry; this does not affect its rank-based AUROC/AP, but future reports should keep the distinction explicit.

## C. Result Existence and Exact Numbers: PASS

All requested artifacts exist. Independent checks found:

- Four run `execution.json` files with `status=DONE`; v4 records rank 2, 10 newly optimized operations including rank selection, 6 reused converged models, and `labels_used=false` (`v4_converged/execution.json:2-7`).
- V4 has 15 saved final model configurations, all converged. The six node/edge-node configurations are explicitly marked reused, while chain/native/rewired continue from frozen v3 parameters (`v4_converged/fitting.json:2-539`).
- The protocol records ranks `[2]`, seeds `[42,43,44]`, the five model kinds, generalized-Bayes head weight, and v3 continuation (`v4_converged/protocol.json:67-100`).
- V4 exact metrics are native AUROC/AP `0.563584835856735 / 0.11923935718358328`, chain `0.5593931271333739 / 0.11590795148602719`, rewired `0.5650653711110646 / 0.11989941042631365`, source-local `0.6472926657312978 / 0.23353247549930678`, source-unit `0.6714761376248615 / 0.18343217532187167`, and legacy-fixed `0.6701903545326996 / 0.1686151292483614` (`v4_converged/results.json:19-45`, `:75-143`). These match `README.md:3`, `RESULTS.md:20-35`, and `RESULTS.json` exactly at full stored precision.
- The native-vs-chain, native-vs-rewired, and native-vs-edge-node AUROC intervals all include zero (`v4_converged/results.json:245-265`, `:310-320`; summarized accurately in `RESULTS.md:39-51`).
- The null has 63/159 case-seed instances above its declared 25% strength gate and mean reassigned mass 0.246654 (`v4_converged/results.json:2129-2135`; `RESULTS.md:72-78`).
- The tracker truthfully records PJ001-PJ003 done, PJ004's statistical controls done but physical replay unfinished, and PJ005 `NOT RUN` (`EXPERIMENT_TRACKER.md:5-15`).

Artifact verification also passed:

- The supplied numeric audit reports 488 verified frozen/protocol hashes, 131 AUROC/AP metric pairs, 228 exact posterior replays with maximum error 0, monotone MAP traces, and valid continuation binding (`NUMERIC_AUDIT.json:2-37`). I reran that audit successfully from the saved artifacts.
- All four `frozen_scores.json` manifests verified: 60 entries per run, no missing paths, no hash mismatches. All four protocol hash sets also verified: 60, 62, 63, and 63 entries for v1-v4.
- The post-run raw manifest openly declares its timing (`provenance_joint_state_20261008_raw_artifact_manifest.json:2-5`). All 159 listed raw measurement files, totaling 7,861,316,942 bytes, independently matched their SHA-256 digests. The score-driving `observations.npz` files were separately bound in each pre-fit protocol.
- All 17 dataset annotations were reconstructed independently and match all four saved annotation files byte-for-byte as structured JSON: 2,968 valid tokens, 265 labeled tokens, and 16 onsets.
- `RESULTS.json` exactly matches every included execution, metric, comparison, seed, within-answer, and null-strength object in the four raw `results.json` files.
- The current six science tests passed (`6 passed`), consistent with `VERIFICATION.json:2-23`. The current `run.py` is a post-run reuse/continuation refactor; v4's executed `run.py` is retained in its code snapshot, and frozen posteriors reproduce unchanged.

One provenance-hardening gap remains: `results.json` and `evaluation_annotations.json` are not members of `frozen_scores.json`, because that manifest intentionally freezes scores before labels are loaded. This audit hashes them directly and independently reproduces them. A separate post-evaluation manifest would make this binding automatic in future runs.

## D. Metric Dead Code: PASS

Every metric-producing function relevant to the reported claims is called and its output is present:

- `paired_bootstrap`, `onset_metrics`, and the optional supervised diagnostic are called from the target evaluator (`evaluate.py:19-108`, `:140-187`).
- `metrics`, `within_answer_auc`, per-case summaries, raw metrics, and per-seed metrics are all written into `results.json` (`evaluate.py:140-183`).
- The independent `rank_metrics` and `check_metric` paths are called for the aggregate, raw, and per-seed verification (`audit_numeric.py:19-64`, `:67-117`).
- The lower-level `metrics`, `bootstrap`, and case-report functions are called by their corresponding evaluator (`grounded_projection_evaluate.py:57-156`).

No reported metric was traced to an uncalled function. `annotations.py:101-121` defines a reusable label-report helper used elsewhere in the research package, but it is not cited as the source of any result here.

## E. Scope and Seeds: WARN

The actual scope is verified as:

- 24 unlabeled fit answers, 12 unlabeled reference answers, and 17 regression answers; 53 unique source IDs with zero source overlap across the three cohorts; 9,118 measured response tokens total (`RESULTS.md:3-7`; source-disjoint assertions are in `run.py:22-44`).
- Regression evaluation covers 10 QA, 3 Summary, and 4 Data2txt answers from four original generators, but all features are measured by a Llama-3.1-8B-Instruct observer. The documents correctly avoid attributing the signals to the original generators (`RESULTS.md:5`).
- Three seeds `[42,43,44]` are used on the same fixed fit/reference/regression cohort (`v4_converged/protocol.json:70-73`). They are optimization/control seeds, not independent dataset replications.
- The regression roster was previously selected with labels (`v4_converged/inputs.json:135747-135751`), all 53 sources were historically exposed, and version changes were informed by earlier outcomes (`RESULTS.md:7-18`).
- Final native performance is weak as a detector: 1/16 onset hits and alarms on 6/6 normal answers (`v4_converged/results.json:89-101`). It is below source-unit AUROC and does not beat the rewired control.
- PJ005 blind confirmation is not run (`EXPERIMENT_TRACKER.md:11`). The physical endpoint replay gate and exact BPE-matched null are also unfinished (`RESULTS.md:72-78`).

The current narrative is honest: it repeatedly says exploratory, previously exposed, no effective detector, no graph necessity, and no online-warning conclusion. The WARN therefore limits what may be claimed; it does not identify a false scope statement in the current files.

## F. Evaluation Type Classification: PASS

| Component | Classification | Evidence and claim ceiling |
|---|---|---|
| Final AUROC/AP, onset, alarm, and per-case evaluation | `real_gt` | Dataset-provided RAGTruth spans loaded only after score freeze (`grounded_projection_evaluate.py:39-54`; `evaluate.py:111-139`). Supports descriptive performance on this saved exposed cohort only. |
| Unlabeled feature fit and rank selection | `self_supervised_proxy` | Predicts observer-derived source gaps without natural labels (`run.py:55-91`; `METHOD_IMPLEMENTATION.md:45-55`). Supports proxy-prediction/model-fitting statements, not factual correctness. |
| Source-blocking, replay agreement, and native-gradient canary | `self_supervised_proxy` | Measures observer-model behavior without factual labels (`measure.py:46-154`; `canary.py:13-60`). Supports measurement fidelity at checked points, not detector efficacy or causal mediation. |
| Synthetic Gaussian and tiny-model science tests | `simulation_only` | Contract/unit tests in `test_science.py:13-117`. Supports implementation contracts only. |
| Supervised logistic diagnostic | `real_gt` (diagnostic only) | Uses dataset labels and is isolated from detector weights (`evaluate.py:62-108`; `supervised_diagnostic.json:2-7`). |

No `synthetic_proxy` reference is used as final ground truth, and no new `human_eval` was performed.

## Action Items

1. Run PJ005 once on a preregistered, previously unexposed, source-disjoint cohort with v4 parameters, score direction, thresholds, and comparison set frozen before annotations are joined.
2. Treat `[42,43,44]` as optimization/control seeds in every claim; add independent data cohorts or model replications before using robustness or generalization language.
3. Complete the physical endpoint replay validity gate and use a stronger BPE/content-matched null before making any graph-endpoint or causal-message claim.
4. Add a post-evaluation manifest covering `evaluation_annotations.json`, `results.json`, visualizations, and the exact evaluation-code snapshot. Keep the existing pre-label score freeze as a separate artifact.
5. Preserve the explicit `self_supervised_proxy` label for source-gap prediction and the observer-model limitation.

## Claim Impact

- **Exact four-round numbers and saved-cohort comparisons:** **supported**. All stored values and hashes reproduce.
- **Detector fitting did not read natural labels:** **supported with qualifier**. The main fit path is label-free; the separate diagnostic uses labels, and the development/evaluation roster is historically exposed.
- **Native improves over raw source on this v4 cohort:** **needs qualifier**. The paired interval is positive versus raw source, but native remains below local/unit baselines and alarms on all normal answers.
- **True native edges outperform chain, rewired, or edge-node controls:** **unsupported**. Relevant AUROC intervals include zero, and rewired point performance is slightly higher.
- **An effective or generalizable hallucination detector has been established:** **unsupported**. PJ005 is unrun and the existing cohort is small and exposed.
- **Measurement/replay implementation is numerically faithful:** **partially supported**. Six tests, four native-gradient checks, frozen posterior replay, and hash checks pass; the planned physical endpoint replay gate is incomplete.
- **Online onset warning:** **unsupported**. Full-answer smoothing uses future observations, and native hits only 1/16 onsets.

## Hash Summary

SHA-256 was computed over 124 explicitly audited files. The deterministic aggregate is `sha256:0ae252b0e1b861a6014744956bfc2b175b404a3cc483624be4969d337acd8821`, using sorted `path + NUL + file_sha256 + LF` records. The complete structured hash summary is in `EXPERIMENT_AUDIT.json`.

Key digests:

- RAGTruth `response.jsonl`: `sha256:e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`
- Current nine target scripts: `sha256:1b4c17c109955f48305e0c0f468b13b7d8220dac5ebaea08404342cc7ec17cac`
- Seven target reports/verification files: `sha256:e46d155f45775d6b598535cef547c57031c99c56666468920c8698bb3a593758`
- Raw artifact manifest: `sha256:9e0a6348be52910d8ad79f015e90d9b1d208ea167a6371a11d9924f3c120590c`
- V4 listed artifacts: `sha256:bee8b8b5cfe620af3d9fa126cf38850f1d67cd272f7a1144abad6d83a2823a9f`
- V4 code snapshot: `sha256:22c8097edc7c17e556635aee5fc2c0890d7071af417a8dbedd404fd5580ad1cf`

This verdict is provisional because reviewer independence is same-family and the requested reviewer runtime/model identity could not be independently attested.
