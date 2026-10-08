# Automatic token source-uptake gate

This experiment tests whether the frozen source-program readout transfers from
six training candidate token types to automatically proposed natural candidates.
It fits **zero natural hallucination labels**, but the readout has source-derived
correct-token targets: this is source self-supervision, not strict unsupervised
learning. The original answers come from multiple generators; Llama-3.1-8B is an
observer. Original-answer labels do not label an observer-generated replacement.

For token `t`, the predictor is `P+t-1`; the current token is never an input to
that observation. One causally masked full-answer prefill captures all rows. Four
prefix-only canaries check equivalence. Eight consecutive rows retain all 32
layers, four sites and 4096 physical coordinates. Source heads are also saved.
Rows are stored once; overlapping windows are formed while scoring.

The unchanged readout uses per-coordinate fit-only normalization, learned linear
state weights and adjacent same-coordinate products, then contracts its vector
with a normalized frozen output-embedding row. The products are statistical
interactions, not native transport or a Transformer Jacobian. Each token has its
own window and score; no annotated span is averaged or broadcast.

Native top32 candidates are automatic. The observed original token is added as
a candidate row only. The primary signed score is the best other candidate's
source compatibility minus the observed token's compatibility. The preregistered
auxiliary adds native log probability with coefficient one before that difference.
A separate native-greedy gap is saved, but it has no replacement-token truth
labels. All three seeds and six frozen views are retained; no test-label selection.

The hash-selected cohort has 16 official QA test sources and all six answers for
each. Sources are disjoint from current readout fit/dev and the two old examples.
The corpus/test has historical exposure: this is an exploratory transfer test,
not blind confirmation. Old source-pair/local unit means and raw route are copied
only after exact token alignment, as historical baselines, not new localization.

Run in the existing research environment from the graph root:

```bash
PYTHONPATH=.:teaching/state_audit/src python -m experiments.flow_latent.source_uptake.run \
  --roster /path/to/ROSTER.json --output outputs/source_uptake_run --limit 8
PYTHONPATH=.:teaching/state_audit/src python -m experiments.flow_latent.source_uptake.run \
  --roster /path/to/ROSTER.json --output outputs/source_uptake_run --limit 96
PYTHONPATH=.:teaching/state_audit/src python -m experiments.flow_latent.source_uptake.evaluate \
  --root outputs/source_uptake_run --roster /path/to/ROSTER.json
```

The first command collects a label-free pilot; the second resumes the identical
protocol and freezes every answer's scores before evaluation can open annotations.
An interrupted partial answer is preserved and requires inspection. Protocol
changes are rejected on resume. Model/source text and raw arrays remain local.

The literature comparison and immutable anchor/plan live under the shared
`lys/codex/research/refine-logs/unsupervised_source_uptake_20261008/` handoff.
FlashTrace supplies span/message attribution, CAGE prompt-ancestry propagation,
ROME native state restoration, and MIRAGE context-dependent attribution. None
alone supplies the truth direction required by hallucination detection. Actual
history-KV intervention is conditional on a reliable source-direction gate; this
capture does not establish a causal source-to-history-to-choice mechanism.

## Measured iteration, 2026-10-08

Complete local output: `outputs/unsupervised_source_uptake_20261008_v2`.
96 answers / 12,853 valid tokens / 501 annotated error tokens; 100 new native
forwards across the pilot and full run, including four prefix-only canaries;
zero new fits. Scores, evaluator/alignment code and settings were frozen before
opening official annotations. All 22 scientific/evaluation/workflow tests pass.

| Fixed seed42 score | AUROC | AP |
| --- | --- | --- |
| Ordered compatibility gap, primary | 0.579375 | 0.049491 |
| Mean compatibility gap | 0.573574 | 0.048191 |
| Ordered compatibility + native logp, secondary | 0.634907 | 0.061035 |
| Native entropy baseline | 0.646939 | 0.086649 |

The primary ordered-minus-mean source bootstrap interval is
[-0.016003, 0.018416]. The auxiliary ordered-minus-mean AUROC interval is positive
[0.006574, 0.030328], but its AP interval crosses zero and overall performance
remains below entropy. No physical transport/graph necessity is established.
The uncalibrated primary gap>0 alarms on 83.7942% of normal tokens and all 81
normal answers. All 15 error answers already alarm earlier in their normal
prefix; first-onset alarms alone are not useful anticipation.

The natural transfer gate failed. 99.11% of observed tokens fall outside the
six-token fit vocabulary; this identifies extrapolation scope, not its causal
responsibility for failure. Native top32 covers 99.60% of original tokens, which
does not establish semantic correctness or correct-rival coverage. No natural
label tuning, default replacement or conditional history-KV intervention followed.
The genuine matched-prefix source-only next-quote task also failed a CPU
feasibility census (3/8 fit and 2/8 reference sources); it was not silently
replaced by artificial controls.

Executed code snapshots and original pilot/reuse records are retained. Final
runner changes after capture fix counter bookkeeping, canonical paths and
automatic metric-protocol freezing; they do not change capture or scoring.
The shared `RESULTS.md` and fresh `EXPERIMENT_AUDIT.md` record the scope and audit.
