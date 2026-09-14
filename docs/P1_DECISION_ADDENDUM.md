# P1 decision clarification (before development labels are joined)

2026-09-13. Scoring code remains SHA256 ab0c13df4bd4ac909e76d50ad19b411c36d3d2d92f9d4e668c4f879ef5374362. No score direction, prompt, model, or primary endpoint changes.

The fixed pilot finished in 86.635 seconds, with median A/B first-token mass 0.9998627. A conservative context-length extrapolation predicts development in 776.033 seconds; the original runtime/format gate passed. The canaries were 7/8: missing_duration was misclassified. This is an explicit semantic failure, not a formatting error, and not repaired by prompt tuning. Development remains a potentially negative test, not a claim that all targeted error types are solved.

For precision before any development-label join:

- Primary score is source_risk. Higher means greater risk. It will not be replaced by the best secondary score.
- Validation may launch only if the primary improves BOTH pooled all-token AUROC and pooled all-token AP over EACH of native NLL, native entropy, and native negative logit margin on the fixed development split. This AP requirement is a conservative tightening of the original development gate, not relaxation after observing natural scores.
- A positive validation claim requires both improvements and paired source-cluster 95% intervals above zero against all three controls, using 500 resamples, seed 20260913. Source-balanced, task-level, onset, and through-first-error endpoints are mandatory diagnostics, not substitute primary endpoints. Source-balanced disagreement or wide intervals must be reported.
- No absolute 'high AUROC', universal-mechanism, or independent official-test claim follows from beating weak controls on these 16 validation sources. Report actual effect sizes, prevalence, and uncertainty.
- The history arm intentionally changes both available evidence and its interpretation. Its difference from the source arm is a diagnostic policy contrast, NOT a clean causal estimate of source use. 'Causal' in the frozen protocol refers only to no-future-prefix access. It does not imply a causal treatment effect or native circuit discovery.
- The verifier score is available after seeing the current token. Native entropy/margin come from the pre-token state. Retrospective sentence scores see future tokens and cannot substitute for either.
- Any later candidate prompted by these development results must get a distinct ID, new immutable outputs, and an explicit developmental-selection history. Failed P1 scores will not be overwritten or re-signed.

The audit is same-family/provisional. The run witness and subsequent result audit are recorded separately. The canary failure remains a limitation even if natural token metrics improve.
