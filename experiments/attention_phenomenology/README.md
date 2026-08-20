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

## Dirichlet suitability audit

A Dirichlet model is not assumed to be correct merely because attention rows lie
on a simplex. `validate-distributions` first checks that assumption against a
logistic-normal alternative on an unlabeled held-out split.

`compositions.py` exposes two representations for every
`[token, layer, head]` row:

```text
role:
  prompt | response_history | self | unresolved

provenance:
  direct_prompt
  | grounded_response_lower
  | unsupported_response_lower
  | uncertain_response
  | self
  | unresolved
```

The second representation uses exact response endpoints and ordered-layer prompt
provenance. It is therefore the first gate for asking whether graph-derived,
multi-hop ancestry adds structure beyond direct role mass.

For each task, causal-position bucket, and layer, `distributions.py` fits:

- one Dirichlet distribution by maximum likelihood;
- one logistic-normal distribution in additive-log-ratio coordinates.

The held-out report does not use hallucination labels. It records:

- held-out log likelihood and train-fit AIC per row;
- Dirichlet minus logistic-normal log likelihood;
- empirical-versus-implied mean and covariance error;
- positive off-diagonal covariance, which one Dirichlet cannot represent;
- dispersion of per-component method-of-moments concentration estimates;
- a simulation-based NLL probability-integral-transform calibration test;
- sensitivity to several zero-replacement pseudocounts.

A single Dirichlet should not be promoted to the detector when it is consistently
outperformed by the logistic-normal model, when empirical positive covariance is
substantial, or when calibration changes materially with the pseudocount. In
that case the next candidate is a mixture, logistic-normal, or graph-conditioned
composition model rather than forcing the Dirichlet likelihood.

Smoke test:

```bash
ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/attention_phenomenology/outputs/dirichlet_smoke \
FIT_LIMIT=20 VALIDATION_LIMIT=10 \
FIT_RESERVOIR_ROWS=128 VALIDATION_RESERVOIR_ROWS=128 \
MINIMUM_GROUP_ROWS=16 SIMULATION_ROWS=256 \
DEVICE=cpu \
  bash experiments/attention_phenomenology/run_distribution_validation.sh
```

The outputs are:

```text
reference.json       fitted Dirichlet and logistic-normal parameters
group_metrics.csv    task / position / layer adequacy diagnostics
summary.json          representation- and pseudocount-level summary
```

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
compositions.py  role and provenance simplex representations
distributions.py Dirichlet/logistic-normal fitting and adequacy metrics
distribution_validation.py label-free held-out distribution comparison
majorization.py  majorization curves and Rényi--Hill diversity spectrum
majorization_dynamics.py streamed exact-source trace and causal state filter
majorization_detector.py label-free robust fit/score interface
majorization_nulls.py registered weight, identity, and chronology controls
majorization_validation.py frozen scoring and post-freeze hypothesis evaluation
head_resolved.py per-token/layer/head routes and cumulative RR-source reuse
head_effects.py independent-validation layer-by-head effect maps
causal_head_model.py small ordered-layer encoder and causal token GRU
head_model_experiment.py source-disjoint train/validation/test workflow
nulls.py         endpoint-rewiring control
experiment.py    fit and score artifacts; never opens labels
evaluation.py    label-aware metrics after score freeze
main.py          CLI only
run.sh           one-command fit -> score -> evaluate workflow
run_majorization_validation.sh one-command majorization validation workflow
run_head_model.sh one-command head-resolved training and evaluation
```

## Majorization and dynamic-state validation

`validate-majorization` tests a more specific token-level mechanism without a
Dirichlet likelihood. For each layer/head independently, it subtracts the
cache floor, keeps exact prompt source IDs, and compares the current sorted
cumulative source curve with a causal EWMA reference. A positive score is
recorded only when the current distribution actually majorizes the reference;
crossing curves receive negative evidence.

The same rows produce the Rényi--Hill effective-source spectrum at orders
`1/2, 1, 2, 4, infinity` and exact-source Hellinger affinity to the preceding
token. A three-state causal filter separates a new concentrated entry from a
stable concentrated residence. `current_probability[t]` is post-emission;
`forecast_probability[t]` is evaluated only against future labels and never
uses token `t+1` attention.

Local smoke test:

```bash
ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/attention_phenomenology/outputs/majorization_smoke \
FIT_LIMIT=3 TEST_LIMIT=2 BOOTSTRAP_REPLICATES=20 DEVICE=cpu \
PYTHON=python bash experiments/attention_phenomenology/run_majorization_validation.sh
```

The workflow writes `reference.json`, freezes all token rows in `scores.npz`,
checks held-out source groups and manifest identity, and only then opens labels
to create `evaluation.json`. It reports current-token detection separately from
1/2/4-token forecasts and records onset and within-span residence effects.
Three registered controls isolate the proposed mechanism: uniform prompt-edge
weights remove weight concentration, per-token source permutations remove
cross-token exact identity, and prompt-row time shuffling removes chronology.

## Head-resolved layer/temporal model

`train-head-model` is the active supervised mechanism test. It never averages
attention heads. Its input is
`[response token, layer, head, feature]`; fixed head coordinates are flattened
inside each layer, an ordered one-way layer GRU produces the token state, and a
one-way temporal GRU emits current-token and next-token logits. Appending future
tokens cannot alter an existing prefix score.

The input includes direct prompt/RR/self/unresolved mass, exact-source
effective count/top-1/velocity, RR lag, history-edge fraction, and
`response_reuse_rank_1..K`. The last fields are the interpretable quantity that
the historical "spectral residual" actually measured: for each layer/head,
the age-normalized cumulative reuse of earlier response sources. No adjacency
eigenvalue or diagonal subtraction is used in these fields.

The original train split is divided by complete `source_id` groups. Training
sources fit the normalizer and network; disjoint validation sources select the
epoch and produce `validation_head_layer_effects.{csv,npz,png}`; the official
test split is opened once after selection. Every evaluation also reports a
strictly causal raw-token-index control. Set `REUSE_TOP_K=0` with the same
limits and seed for the registered no-reuse ablation. Run locally from Git
Bash:

```bash
ROOT=D:/projects/python_projects/research/data/RAGTruth/llama31_8b \
PYTHON=D:/projects/python_projects/.audit_envs/llm_state_lab_py311/Scripts/python.exe \
DEVICE=cpu TRAIN_LIMIT=128 TEST_LIMIT=64 EPOCHS=12 \
bash experiments/attention_phenomenology/run_head_model.sh
```

For the matched no-reuse control, change only the output directory and
`REUSE_TOP_K`:

```bash
OUT=experiments/attention_phenomenology/outputs/head_model_no_reuse \
REUSE_TOP_K=0 ROOT=D:/projects/python_projects/research/data/RAGTruth/llama31_8b \
PYTHON=D:/projects/python_projects/.audit_envs/llm_state_lab_py311/Scripts/python.exe \
DEVICE=cpu TRAIN_LIMIT=128 TEST_LIMIT=64 EPOCHS=12 \
bash experiments/attention_phenomenology/run_head_model.sh
```

The logistic-normal comparison has a narrower purpose. It established that a
single Dirichlet cannot describe the covariance of role/provenance
compositions; logistic-normal is a better nuisance/reference distribution.
That does not itself distinguish hallucination. A logistic-normal NLL may be
tested later as a covariance-aware baseline, but it is not an input to the
head-resolved model and is not presented as a mechanism discovery.

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
