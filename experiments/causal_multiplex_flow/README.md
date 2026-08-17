# Causal Multiplex Routing Prediction

CMRP is a label-free, source-aware dynamic attention-graph experiment.  It
learns to predict retained response-history source identities from the causal
multiplex graph state that exists before each response token.

Read in order:

1. [`METHOD.md`](METHOD.md): scientific design and claim boundaries;
2. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): module and validation
   plan;
3. this file: commands and artifact interpretation.

## What this implementation preserves

- exact RR source and target token identity;
- response generation order;
- layer/head channel of every selected retained edge;
- retained attention weight;
- RP versus RR edge role;
- multi-hop history through sequential source node states.

It intentionally does **not** use prompt relative-position centroids.  Prompt
sources are represented by the common RP role until a validated semantic prompt
segmentation is available.

## What the primary score means

For token `t`, the model predicts:

1. whether a retained RR edge should exist;
2. the true source of selected retained RR edges among deterministic hard
   negatives, including a lag-bin-preserving rewired source.

The raw routing surprise is:

```text
presence NLL + mean retained-RR source contrastive NLL
```

A disjoint unlabeled calibration source-group split converts this raw value to a
monotone empirical upper-tail score.  Larger values mean the observed retained
routing is less typical under the fitted train process.

This is a contrastive retained-event score, not a normalized probability over
all legal dense attention edges. Missing cache entries remain censored.

## Smoke test

Use a nontrivial train limit so fit and calibration source groups are both
present:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

TRAIN_LIMIT=64 TEST_LIMIT=5 EPOCHS=1 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_multiplex_flow/run.sh
```

Output:

```text
experiments/causal_multiplex_flow/outputs/
smoke_train64_test5/
├── model.pt
├── reference.npz
├── test_scores.npz
└── evaluation.json
```

The smoke result validates runtime only. It is not used to choose model
components or report final performance.

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_multiplex_flow/run.sh
```

Default output:

```text
experiments/causal_multiplex_flow/outputs/full/
```

Common overrides:

```bash
EPOCHS=3 \
MAX_RP_EVENTS=24 \
MAX_RR_EVENTS=48 \
NEGATIVES=12 \
HIDDEN_DIM=96 \
OUT=/absolute/new/output \
bash experiments/causal_multiplex_flow/run.sh
```

Changing these after inspecting final test labels requires a fresh held-out
evaluation split for a clean claim.

## Artifacts

### `model.pt`

Contains only:

- model state;
- attention geometry;
- event/model/train configs;
- no labels.

### `reference.npz`

Contains:

- model digest and config records;
- disjoint fit/calibration group IDs;
- calibration raw routing-surprise distribution;
- label-free topology gate:
  - mean/median true-versus-rewired gap;
  - positive-gap fraction as a diagnostic only;
  - evaluated-edge and selected-edge counts;
  - evaluated/selected coverage;
  - pass iff a finite edge was evaluated and mean edge gap is positive;
- no labels.

A useful topology model should give:

```text
rewired_source_nll - true_source_nll > 0
```

on held-out calibration groups.

### `test_scores.npz`

Contains one row per response token:

```text
sample_id
source_id
token_index
response_length
task_type
data_source
generator_model
score                         # primary calibrated score
raw_route_surprise
presence_nll
source_nll
weight_error
rewired_source_nll
rewire_gap
selected_rr_edges
```

The v2 score schema also stores the complete fit/calibration/test source-group
audit and the exact selected test-sample scope. Scoring rejects an overlapping
test source while each loaded sample is streamed into the scorer; a partial test
run is explicitly recorded. It also stores the SHA-256 of the exact on-disk
test `manifest.json`; every selected response must contain exactly token rows
`0..response_length-1` with one canonical source and response length.

Only `score` uses the automatic score-field convention. Raw diagnostics avoid a
`score_` prefix so the conditioned benchmark cannot silently treat them as
independently selected detectors.

### `evaluation.json`

Labels are opened only after the score artifact digest and exact on-disk dataset
manifest digest are reverified. Before label access, evaluation also checks the
`test` split, complete token coverage, canonical source, and attention-derived
response length. It reports AUROC/AUPRC for the frozen primary score and
post-hoc diagnostic components.

## Common conditioned benchmark

After both CMRP and the RR spectral method have frozen score artifacts:

```bash
bash experiments/conditioned_benchmark/run.sh \
  /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/test \
  experiments/conditioned_benchmark/outputs/cmrp_vs_rr \
  cmrp=experiments/causal_multiplex_flow/outputs/full/test_scores.npz \
  rr_spectral=experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2/full/test_scores.npz
```

The benchmark intersects methods on identical `(sample_id, token_index)` rows
and evaluates shared task/prevalence conditions without refitting either method.

## Interpretation gates

Do not present CMRP as a successful topology detector unless:

1. the unlabeled calibration topology gate prefers true sources over
   lag-preserving rewires;
2. the frozen primary score beats role-mass and RR spectral baselines on a fresh
   held-out evaluation;
3. removing source state or layer/head identity degrades performance;
4. fit/calibration/test source groups and label access remain disjoint.
