# Teacher-forced three-mechanism audit

This experiment runs one frozen Llama-3.1-8B observer over every cached
RAGTruth response. `train` and `test` are only physical cache locations. They
are pooled within each task, while QA, Summary, and Data2txt are evaluated and
reported separately.

## Core calculation

For response token `t`, visible source token `s`, layer `l`, and query head
`h`, the functional-message magnitude is

```text
e[l,h,t,s] = A[l,h,q_t,s] ||W_O[l,h] V[l,g(h),s]||_2
```

`q_t` predicts token `t`, and `g(h)` maps a query head to its GQA KV head. The
calculation therefore uses the sample's actual attention weight, value vector,
and matching layer/head block of `W_O`; heads are not averaged first.

External evidence means the passages for QA, the source article for Summary,
and the structured record for Data2txt. Four teacher-forced branches are run:

| branch | messages deleted after softmax |
|---|---|
| `full` | none |
| `no_evidence` | external-evidence messages |
| `no_response` | visible response messages |
| `no_evidence_response` | both groups |

Deleted mass is not renormalized. The hidden state continues through the real
output projection, residual path, MLP, and later layers. The fixed token scores
are

```text
causal_route_capture = logp_no_evidence - logp_no_response
routing_imbalance = response_message_share - evidence_message_share
source_dispersion = mean_l entropy_s(sum_h e[l,h,t,s]) / log(visible_sources)
message_independent_preference = margin_no_evidence_response
```

Only compact layer/token statistics, top message sources, and the four branch
inputs are saved. Dense attention, value tensors, hidden states, and checkpoint
weight copies are not written to disk. Hallucination labels remain sealed until
the fixed scores have been constructed; they are then used only for post-hoc
AUROC and AUPRC.

The task report also retains the mechanism-difference audit used before the
fixed detector was introduced. For evidence share, response share, routing
imbalance, and source dispersion it reports the all-layer mean, early third,
late third, and late-minus-early shift. Hallucinated and correct tokens are
compared only inside the same response and the same absolute/relative position
cell, then aggregated with equal source weight and source-level bootstrap
confidence intervals. These label-dependent contrasts diagnose the mechanism;
they never select, fit, weight, or reverse a detection score.

## Run all data

From the repository root, run the single foreground command:

```bash
bash experiments/attention_mechanism_audit/run_all.sh
```

The shell file contains only:

```bash
python -m experiments.attention_mechanism_audit.run all
```

The default output root is
`experiments/attention_mechanism_audit/outputs/Meta-Llama-3.1-8B-Instruct/`.
`traces/train/` and `traces/test/` are the shared resumable captures. Each of
`qa/`, `summary/`, and `data2txt/` contains only its own pooled `report.json`,
`token_scores.npz`, and population `figures/`. Changing the bootstrap seed
recomputes confidence intervals without changing or duplicating the captured
traces. The all-data run does not create per-sample figures.

Optional path overrides are available from
`python -m experiments.attention_mechanism_audit.run all --help`.

## Plot one sample

Render a saved sample by ID without replaying the model:

```bash
python -m experiments.attention_mechanism_audit.run plot-sample \
  --input /path/to/output/traces/train \
  --input /path/to/output/traces/test \
  --sample-id SAMPLE_ID \
  --output sample.png
```

The audit describes how the frozen observer processes teacher-forced tokens.
If another model generated the cached answer, it does not recover that
generator's internal process.

Run the focused tests with:

```bash
pytest -q experiments/attention_mechanism_audit/tests
```
