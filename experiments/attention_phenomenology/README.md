# Attention Phenomenology Audit

This experiment tests one token-level routing hypothesis:

```text
prompt access weakens
    -> heads disagree about prompt versus response routing
    -> response history takes over
    -> heads repeatedly use a small set of exact response sources
```

The result is an attention-routing audit, not proof of hidden-state conflict or
causal influence on logits. Feature extraction and reference fitting do not read
hallucination labels. Labels are opened only by `evaluate`.

## Representation

For every response token, layer, and head, `routing.py` keeps four masses:

```text
prompt | response_history | self | unresolved
```

`unresolved` is attention mass absent from the thresholded cache. Exact prompt
and response source IDs remain on the sparse edges; they are not replaced by
position or lag bins and are not summed across heads. Response lag is used only
for recency summaries and for the endpoint-rewiring control.

`sources.py` computes exact-source fields for each `[token, layer, head]` row:

- effective number of sources;
- strongest-source share and exact source ID;
- Hellinger velocity from the previous token;
- recent response share and mean response lag.

No dense `[token, layer, head, source]` tensor is constructed.

## Features

`features.py` returns `[token, layer, 21]` in four mechanism families:

| Family | Features |
|---|---|
| access | prompt mass, effective sources, top-1 share, source velocity |
| fracture | prompt/response head disagreement, prompt-mass dispersion, provenance dispersion, exact prompt-anchor agreement |
| integration | prompt-provenance lower bound, censoring uncertainty, unsupported response mass |
| lock-in | response takeover, effective sources, top-1 share, recent share, mean lag, source velocity, exact response-anchor agreement |
| controls | self, unresolved, and known mass |

Head disagreement is the mean pairwise Hellinger distance between each head's
two-way `[prompt, response_history]` distribution. Anchor agreement is the
fraction of valid head pairs whose strongest exact source token is identical.
These replace the previous persistence-homology, effective-rank, LID, random
projection, and Top-K-turnover bundle.

`provenance.py` separately propagates prompt ancestry through ordered layers.
The lower bound uses observed prompt, self, and response-source routes. The upper
bound additionally allows unresolved mass to be prompt-derived. Because self
attention participates in this recurrence, `self_mass_mean` is always reported
as an explicit control.

## Scoring and tests

`reference.py` fits train-only medians and robust scales conditioned on task and
causal log-position. Standardized values have an explicit scale floor and are
clipped before equally weighted RMS family aggregation. No PPCA, learned fusion,
or label-derived weight is used.

At evaluation time the audit asks:

1. Are family scores atypical on hallucinated tokens?
2. Do the registered feature directions change at first onset?
3. Does a fractured state consolidate later in the same span?
4. Does exact endpoint identity add signal beyond prompt/response role,
   prompt-position bin, and response-lag bin?

The fourth question uses `nulls.py`, which rewires exact sources while preserving
target, layer, head, weight, role, prompt-position bin, and response-lag bin.

## Execution path

```text
main.py
  -> experiment.fit_reference()
  -> experiment.score_split()
  -> evaluation.evaluate_scores()
```

Core files have one responsibility:

```text
routing.py       sparse cache -> four-role routing state
sources.py       per-head exact-source statistics
provenance.py    ordered-layer prompt ancestry bounds
features.py      named mechanism features
hypotheses.py    feature families and predicted directions
reference.py     unlabeled position/task reference and family scores
nulls.py         endpoint-rewiring control
experiment.py    fit and score artifacts; never opens labels
evaluation.py    label-aware metrics after score freeze
main.py          CLI only
run.sh           one-command fit -> score -> evaluate workflow
```

## Run

Smoke test:

```bash
ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/attention_phenomenology/outputs/v3_smoke \
TRAIN_LIMIT=20 TEST_LIMIT=5 BOOTSTRAP_REPLICATES=20 \
DEVICE=cpu bash experiments/attention_phenomenology/run.sh
```

Full run: omit `TRAIN_LIMIT` and `TEST_LIMIT`, and use a fresh `OUT` directory.
Artifacts use the `attention-phenomenology-*-v3` schemas; v1/v2 references and
scores are intentionally incompatible with this representation.
