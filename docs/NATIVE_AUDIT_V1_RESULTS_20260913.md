# Native v1 natural-data result and next iteration

Recorded 2026-09-13T05:36:30.868925+08:00. Development batch: 36 responses from six official train sources (two per task, six generators per source). This is not the full RAGTruth population and not a test-set result. The observer is Llama3.1 replaying responses; Qwen reader predictions are not ground truth.

All A/B/C/D/merge predictions completed before the independent annotation join. Official response annotations were verified against the pre-run evaluation manifest, exact identities and response hashes. Evaluation: outputs/native_audit_v1_20260913/evaluation.json. Executed sources are preserved under executed_code/; current source has begun v2 and must not be substituted when reproducing v1.

| Quantity | Result |
|---|---:|
| All response words | 4733 |
| Annotated error words | 260 |
| Questions: invalid / uncertain / supported / unsupported | 103 / 56 / 13 / 0 |
| Semantic scored words | 287 (6.0638%) |
| Annotated error words with a semantic score | 3 (1.1538%) |
| Error words detected at frozen threshold 0.8 | 0 |
| Recall including abstentions | 0 |
| Actual native forwards / processed native tokens | 0 / 0 |
| Mechanism coverage | 0 |

The all-word source-balanced semantic AUROC is 0.525407 and AUPRC 0.057373; mechanism ranking is the abstention baseline (AUROC 0.5). These do not establish a usable detector. Thirteen supported predictions are reader estimates, not thirteen independently verified correct answers. Zero risky windows means this run provides no test of the internal graph's ability to distinguish constraint ownership. It neither establishes nor refutes sufficiency of internal information once valid natural decision windows exist.

Failure diagnosis before annotation join identified broad paragraph/answer extraction: 75 self-QA answer mismatches, 17 hidden-answer premise overlaps, 7 absent/ambiguous quotes; 19 source/verification invalid-JSON calls and 9 citation failures. These categories can overlap. The new design was motivated by these pre-label front-end failures, not by selecting favorable labeled examples or changing thresholds. The same six-source roster is now explicitly development data. Any later generalization claim requires new untouched sources.

V2 replaces freeform question generation with atomic event-role extraction, exact surface/antecedent bindings and deterministic leave-one-role-out questions. Every remaining role must have an exact separate condition-table entry, with source quotations tied to the same event. Missing non-target premises cause uncertainty, including for target-absence estimates. All earlier contrast/origin/control tests remain. CPU tests do not establish factual accuracy. V2 will rerun this identical full roster into a fresh output and preserve every failure/abstention.

Execution caveat: native audit child exit 0; wrapper exit 1 because its 120-second population restoration check expired during file validation. Original journal remains population_resume_unverified. Subsequent independent observation confirms original population restored as PID141861 and advanced 11515 to11562 with failed0. A new scheduler only increases this verification deadline to900 seconds; actual PID/state/held-lock verification remains required.

Result-to-claim: claim_supported=no for accurate automatic lookback localization, applicable-constraint identification, routing-derived hallucination detection, and continuous-error span recovery. Codex MCP unavailable; this parent judgment is [pending Codex review]. A separate collaboration reviewer is checking integrity, explicitly not impersonating that backend.
