# Prompt-carrier route-collapse detector

This experiment implements the ICLR audit conclusion directly: hallucinated
tokens can be associated with attention concentrating on fewer prompt carriers,
cross-head routing losing independent modes, and the same carriers persisting
over adjacent response tokens. QA, Summary, and Data2txt are evaluated separately.

## Two exact ledgers

For response predictor t, prompt source s, layer l, and query head h:

    attention route = A[l,h,t,s]
    functional edge = A[l,h,t,s] ||W_O[l,h] V[l,g(h),s]||_2

The first ledger preserves the earlier attention result. The second verifies
that concentration survives the dynamic value and matching layer/head output
projection. GQA query-to-KV mapping is explicit. Neither ledger averages heads
before measuring cross-head structure.

For both ledgers the capture saves effective prompt sources in the cross-head
mixture, mean per-head prompt entropy and top-1 share, cross-head JSD,
participation-ratio effective rank of the head-by-prompt route matrix, and every
head's dominant prompt-carrier identity.

Evidence, other prompt, strict response history, and predictor self remain
separate. Four symmetric causal branches remain explanatory audits, not
substitutes for carrier collapse.

## One formal detector

At every layer the route-volume is the product of effective prompt sources,
effective head-route rank, and effective prompt anchors over the current and
previous three tokens. This is the number of effective routing degrees of
freedom across source, head, and recent time. Its log is compared with the
expected log-volume for the same task, response position, and response length.
Only a lower-than-expected volume contributes to the hallucination score.

Fit, calibration, and held-out sources are disjoint. Fit sources estimate the
position/length baseline; separate unlabeled calibration sources convert the
one-sided collapse into an empirical percentile. Labels are opened only after
all out-of-fold scores are frozen.

The primary score is functional_route_collapse. The attention-only
attention_route_collapse and token surprisal are controls. There is no PPCA, AR
transition model, reconstruction detector, label-selected direction, or second
fallback implementation.

The post-hoc audit reports effective carriers, effective rank, top-1 share,
head disagreement, anchor turnover, role routing, exact write coherence, and
factorial evidence/history effects with matched source-cluster intervals.

## Files and schema

- capture.py: dynamic routing/message ledgers and causal replay.
- collect.py: traversal, identity, resumable serialization; no labels.
- detect.py: source-cross-fitted route-volume scoring; no labels.
- evaluate.py: label-opened AUROC/AP and mechanism audit.
- run.py: the single foreground CLI.

Schema 6 writes fresh artifacts to routing_state/train and routing_state/test;
incompatible earlier caches are never silently reused.

Run all tasks in the foreground:

    bash experiments/attention_mechanism_audit/run_all.sh

The shell file contains only:

    python -m experiments.attention_mechanism_audit.run all

Run tests:

    python -m pytest -q experiments/attention_mechanism_audit/tests
