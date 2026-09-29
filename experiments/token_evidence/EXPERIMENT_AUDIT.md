## Experiment integrity audit

**Overall verdict: WARN**
**Review independence:** same-family
**Acceptance status:** provisional

No fake ground truth, metric inflation, phantom result, or hidden label use was found. The experiment completed and the negative result is reported honestly. Any claim that `reset_cad_tail` improves detection is **unsupported**.

I independently confirmed:

- All 74 answer archives contain finite, correctly shaped arrays with matching token IDs.
- The totals match: 13,941 tokens and 5,766 model calls.
- All thresholds reproduce within floating-point tolerance, with a maximum difference below `4.1e-7`.
- All 90 cohort/method summary blocks reproduce exactly.
- All three bootstrap intervals reproduce exactly.
- The cited `Sure`, `gear`, and `aware` examples match `units.csv`.
- `TOKEN_AUDIT.html` contains 54 evaluated answers and correctly shows no token-level GSM alarms.

### A. Ground-truth provenance — PASS with cohort qualifications

Thresholds are computed and written before labels are loaded: `experiments/token_evidence/evaluate.py:44-73`, followed by label access at `:76-79`; `evaluate()` preserves this sequence at `:172-175`.

- Official RAGTruth evaluation is `real_gt`: dataset labels are loaded and checked for exact response-text and annotation coverage at `experiments/native_support/ragtruth_benchmark/data.py:111-137`.
- Local natural pairs are `human_eval` diagnostic labels. `experiments/role_free_flow/diagnostics.py:27-36` converts analyst-designated supported/unsupported spans into labels. `experiments/path_conflict/paired_cases.json:5,25-27,39` explicitly limits them to reviewed source support rather than official or exhaustive world truth.
- GSM is `real_gt` only at step level. `experiments/gsm8k_recurrence/evaluate.py:13-16` labels pre-error steps negative, the first error positive, and later steps unknown.
- No model output is used as a reference target. The detector is a model-relative self-supervised proxy, defined at `experiments/token_evidence/readout.py:7-32`.

A latent roster bug remains at `experiments/token_evidence/prepare.py:35-36`: the selected minimum-hash response is assigned `original.key=candidates[0]["id"]`. All 36 current held-out records happened to have matching IDs, so this run is unaffected. Future multi-answer source/generator packs could mis-key labels. Set the selected row’s key from its own `row["id"]`.

### B. Metric normalization — PASS

The new features use legitimate probability normalization through `log_softmax` and `logsumexp` at `readout.py:7-32`. No final metric is divided by a maximum, mean, or another prediction-derived statistic.

Task thresholds are source/answer-equal weighted empirical 95th percentiles at `evaluate.py:27-30,60-72`. AUROC and AP consume the raw scores at `:103-118`.

The historical base uses reference-rank normalized features, not test-self-normalized metrics. It is also an offline comparator: its route feature uses `window_mean(..., offline=True)` at `experiments/probabilistic_detection/data.py:53-65`, and `fixed_unsupervised` combines unit/source and route ranks at `experiments/unsupervised_graph/scalar.py:17-25`. The results disclose this at `RESULTS_ZH.md:36`, but naming it `offline_base` would make the comparison clearer.

### C. Result existence and scope — PASS; efficacy result is negative

`capture_complete.json:2-24` records 74 answers, 13,941 tokens, 5,766 calls, and `labels_used=false`. `evaluation_complete.json:2-24` records 7,493 evaluated rows and the retrospective scope. Capture, evaluation, and report all exited successfully according to `EXECUTION.json:16-19`.

The primary results are:

- Historical official RAG: `0.470298` versus base `0.808717` (`summary.json:3-19,615-631`).
- Held-out QA: `0.502443`, 0 TP/83 FP versus base `0.998704`, 59 TP/26 FP (`summary.json:2529-2545,3357-3373`).
- Held-out Summary: `0.587585` versus `0.843950` (`summary.json:3911-3927,4739-4755`).
- Held-out Data2txt: `0.612217` versus `0.902972` (`summary.json:5293-5309,6121-6137`).
- GSM: three positive steps only; primary 1 TP/1 FP (`summary.json:1687-1703,2191-2207`).

All three source-cluster AUROC-difference intervals are negative (`bootstrap.json:2-34`). The written conclusion correctly calls the method a failure and declines to replace the default (`RESULTS_ZH.md:1-3,28-32`).

The 36 held-out answers are source-disjoint from this run’s case/dev records, but come from a historically exposed official test partition. This is disclosed at `prepare.py:26-40,52`, `evaluation_complete.json:24`, and `RESULTS_ZH.md:11,28`. They do not support fresh independent validation.

### D. Called versus dead evaluation code — PASS

The live path is complete:

- `run.py:11-20` dispatches preparation, capture, and evaluation.
- Capture calls both full-vocabulary score functions.
- Evaluation calls freezing, label construction, summaries, localization, within-unit metrics, and bootstrap at `evaluate.py:172-197`.
- Reporting is now documented at `README.md:13-17`, executed successfully, and produced `COMPARISON.md` and `TOKEN_AUDIT.html`.

The supervised `threshold_at_fpr` path in `probabilistic_detection/evaluation.py:26-29,59-69` is not imported or called by this experiment. It did not affect thresholds.

### E. Source isolation, cache semantics, leakage, and evaluation correctness — WARN

The reset implementation is semantically correct for the configured Llama model:

- `native.py:15-18` aligns prompt/answer prediction states and crops to clean prompt KV.
- `native.py:30-39` reconstructs `[prompt[-1]] + answer[t-L:t]` with original RoPE positions.
- `native.py:21-24` creates independent branch caches.
- `native.py:43-47` batches only equal-length suffixes.
- No future answer token enters either full or reset prediction.

The full-vocabulary tails are correct. `readout.py:7-13` sums all less-likely tokens plus half of ties; `:16-32` applies the corresponding likelihood-ratio tail. Top-eight candidates at `:33-38` are explanation-only.

Three warnings remain:

1. **First-token residual path.** Source masking removes attention keys but cannot remove the last prompt token’s own embedding/residual from the state predicting answer token 0 (`capture.py:25-30`; `native.py:12-15`). Fourteen GSM records have the final prompt position marked as source. The report accurately discloses this at `RESULTS_ZH.md:41`. If full source removal is intended, use a fixed unmasked separator or flag/exclude token 0, and add a test changing a masked final prompt token.

2. **Executed versus current capture checks.** The executed snapshot checked replay only when `targets[-1] <= window`, skipping `t=16`, and did not explicitly reject NaNs (`executed_capture_code/capture.py:41-48`). Current code fixes both at `capture.py:41-52`. Post-run verification found finite arrays and first-17 score/log-prob differences below `1.42e-4` (`artifact_verification.json:5-20`), but this does not retroactively provide a full-vocabulary runtime replay check at `t=16`. The report states that boundary correctly at `RESULTS_ZH.md:53-55`.

3. **Unknown-region localization.** `evaluate.py:125-145` forms alarms over all positions, including `gold=-1`, then uses their minimum as `predicted_first`. In the completed data, 464/495 local rows and 20/44 GSM rows are unknown. For the local cohort, every primary `predicted_first` lies in an unknown position. This also changes `first_alarm_exact_hits` for `full_cad_tail` and `full_cad_nll` from 0 to 1 when restricted to certified positions. Core AUROC/AP/TP/FP and the reported first-error-token hit counts are unaffected because those use known labels. Fix certified localization by constructing alarms from `known`; report whole-answer alarms separately as censored descriptive output.

The freeze order is correct, but freeze and label evaluation occur in one rerunnable process and no executed evaluator/import hash manifest is stored. Split these into immutable stages or hash the evaluator, imported helpers, manifest, score archives, and thresholds before label access.

### F. Evaluation classification

| Component | Classification |
|---|---|
| New score and unlabeled-dev threshold | `self_supervised_proxy` |
| Official RAGTruth cases and held-out subset | `real_gt`, dataset-provided human annotations |
| Local natural pairs | `human_eval` diagnostic proxy, restricted to reviewed spans |
| ProcessBench GSM | `real_gt` at step level; later steps unknown |
| Historical base | Unsupervised offline proxy with unit/future-window aggregation |

### Remaining provenance gaps

The claim that five tiny-model tests passed appears at `RESULTS_ZH.md:53`, while `EXECUTION.json:6` records only the count, without durable test stdout or a test exit code. The tests themselves are well targeted, but their execution remains unverified from persisted artifacts. Save a test log and exit status in future runs.

No main metric remains numerically unverified. I did not exhaustively verify every decoded candidate string in the 1.2 MB HTML report or the full-vocabulary logits at the uninstrumented `t=16` boundary.

No files were modified, and I ran no model or GPU workload.

## Post-fix reviewer addendum

Addendum: both requested defects are fixed.

- **Unknown-label localization: fixed.** `evaluate.py:127-146` now derives certified alarms and `predicted_first` only from `gold >= 0` rows, while retaining `predicted_first_all_positions` as censored descriptive output. The regression test at `test_token.py:62-69` passes. `localization_fix_verification.json:2-20` confirms thresholds, scores, and core metrics are unchanged; only the two expected local `first_alarm_exact_hits` values change from 0 to 1. The original evaluation is preserved under `evaluation_before_localization_fix/`.
- **Selected-row key: fixed.** `prepare.py:35-36` now assigns both the selected response and its key from `selected`, eliminating the prior `candidates[0]` mismatch risk. There is no dedicated regression test for this selection case.
- **Test provenance: resolved.** `tests.log:1-11` records all six tests passing.

The overall audit remains **WARN / same-family provisional** because the other limitations remain: GSM token-0 retains the final masked prompt residual, the executed capture lacked the full-logit `t=16` runtime check, the base is offline, the held-out subset is historically exposed, and freeze/evaluation provenance is not hash-locked.
