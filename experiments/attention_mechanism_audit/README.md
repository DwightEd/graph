# Teacher-forced three-mechanism audit

This experiment asks whether hallucinated response tokens show three specific
attention-message mechanisms in a frozen Llama-3.1-8B observer:

1. **routing imbalance**: functional message mass shifts from the supplied QA
   evidence toward the generated response history;
2. **source dispersion**: evidence routes spread over more source tokens;
3. **message-independent preference**: the observed token remains preferred
   after evidence and response attention messages are removed.

It processes every QA response available in the formal attention archive.
The archive happens to store files under `train` and `test`, but those names
are storage locations only: this experiment trains nothing, evaluates no
split separately, and pools both locations into one result.

Directory-level implementation constraints are recorded in `AGENTS.md`, so a
coding agent working in this module receives the same minimal-code and exact-
method contract instead of relying on rules stored in a sibling experiment.

## Core calculation

For response token `y_t`, source token `s`, layer `l`, and attention head `h`,
the dynamic directed message is

```text
m[l,h,t,s] = A[l,h,q_t,s] W_O[l,h] V[l,g(h),s]
```

where `q_t` is the predictor position and `g(h)` maps a query head to its GQA
KV head. Its edge magnitude is

```text
e[l,h,t,s] = A[l,h,q_t,s] ||W_O[l,h] V[l,g(h),s]||_2
```

This is not an attention-only proxy: `A` chooses a route, `V` supplies its
sample-specific content, and the head block of `W_O` writes that content into
the residual stream.

The frozen model is then replayed with evidence messages or response-history
messages removed after softmax, without renormalizing other routes. Removed
hidden states continue through the real MLP and every later layer. With
`L_full`, `L_no_evidence`, and `L_no_response` denoting observed-token log
probabilities, the four fixed label-free scores are:

```text
causal_route_capture = L_no_evidence - L_no_response
routing_imbalance    = response_message_share - evidence_message_share
source_dispersion    = normalized entropy of source-token message magnitudes
message_independent_preference = margin_no_evidence_or_response_messages
```

`causal_route_capture` is the main score. The other three values are reported
separately as mechanism components; they are not concatenated into a learned
classifier. The preregistered direction of source dispersion is positive; an
AUROC below 0.5 is evidence against that hypothesis and is not flipped after
labels are read.

The final margin is a message-independent preference candidate, not a pure
parameter-knowledge measurement: question/constraint messages, the predictor
residual embedding, and all MLP updates remain active.

Labels are unavailable to capture and to all four score equations. They are
opened only after both physical cache locations have been captured, to compute
pooled token-level AUROC and AUPRC. These are post-hoc detection diagnostics,
not a trained classifier or a calibrated deployment threshold.

## One-click run

From the repository root, first run one cached response as a smoke test:

```bash
LIMIT=1 bash experiments/attention_mechanism_audit/run_qa.sh
```

Then capture/resume and evaluate every cached QA response:

```bash
bash experiments/attention_mechanism_audit/run_qa.sh
```

The script uses the local Meta-Llama-3.1-8B-Instruct checkpoint and formal QA
cache by default. It performs only two logical stages:

1. capture or resume exact traces from the physical `train` and `test` cache
   directories;
2. pool those traces once and produce one all-data evaluation.

It intentionally does not use `set -euo pipefail`. If Python fails, its full
traceback remains visible and the next stage is not run.

The compact outputs are:

- `{train,test}/traces/`: resumable exact captures for the two physical cache
  locations;
- `token_scores.npz`: sample/token identity, labels, the main score, and
  the three mechanism components;
- `report.json`: pooled coverage plus AUROC and AUPRC;
- `figures/`: ROC/PR, mechanism distributions, and relative-position dynamics
  computed from all pooled tokens.

No per-sample figures are generated during the all-data run.

## Inspect one sample on demand

After capture, render one response by sample ID without rerunning the model:

```bash
python -m experiments.attention_mechanism_audit.run plot-sample \
  --input /path/to/train/traces /path/to/cache/train \
  --input /path/to/test/traces /path/to/cache/test \
  --sample-id SAMPLE_ID \
  --output sample.png
```

The same command can be placed in a notebook cell and `SAMPLE_ID` changed
interactively. `sample_explorer.ipynb` already contains both cache locations,
so only `SAMPLE_ID` needs to change. This view is for tracing a concrete case;
the headline plots always come from the pooled all-data evaluation.

## Scope

The audit measures how this frozen observer processes teacher-forced tokens.
If the original response was generated by another model, it does not establish
the original generator's exact formation process. The evidence region is the
QA passage block supplied by the dataset, not a manually annotated minimal
support span.

Run the focused tests with:

```bash
pytest -q experiments/attention_mechanism_audit/tests
```
