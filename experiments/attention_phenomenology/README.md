# Attention Phenomenology Audit

This subproject asks whether cached Transformer attention contains mechanisms
analogous to those reported in **The Phenomenology of Hallucinations**
(arXiv:2603.13911), without pretending that attention weights are hidden-state
activations or output sensitivity.

It is an attention-only, token-level, label-free construction. All attention is
read through `research_dataset.py`. Labels are opened only by the final
`evaluate` command.

## What is and is not being transferred from the paper

The paper studies residual-stream geometry, persistent homology, output-layer
sensitivity, and causal interventions. The current cache contains attention
weights but not residual states, logits, values, output projections, or MLP
updates. Consequently this project tests attention-space analogues only:

| Paper mechanism | Attention-space question | Claim boundary |
|---|---|---|
| Detection | Does the routing state become atypical before or at hallucination onset? | Routing atypicality is not model uncertainty. |
| Topological fracture | Do heads split into separated routing coalitions? | Head-routing fragmentation is not activation fragmentation. |
| Failed integration | Do some heads remain prompt-grounded while others route through unsupported response history? | Equal-head attention composition is a proxy, not actual residual contribution. |
| Amplification / breach | Does an initial route reorganization settle into stable, concentrated, weakly grounded response feedback? | This is routing lock-in, not a logit-space breach. |
| Low-sensitivity output subspace | Not testable from this cache. | Requires residual states and output Jacobians/logits. |

## Pre-registered hypotheses

### H1 — Routing detection

A hallucination onset is preceded by or coincides with a routing state that is
atypical under an unlabeled task- and causal-position-conditioned train
reference.

The project fits only robust medians and MADs on train attention. It reports
separate direction-free atypicality scores for fracture, integration, and
lock-in fields. No hallucination label selects a feature, layer, direction, or
weight.

### H2 — Routing fracture

At or near onset, the 32 heads in some layers split into more separated source
routing coalitions.

For token `t` and layer `l`, each head is represented by a probability vector
on a fixed role simplex:

```text
[prompt-position bins,
 response log2-lag bins,
 self,
 unresolved/censored mass]
```

The unresolved coordinate is essential: an unretained edge means
`attention <= attention_floor`, not an observed zero.

The 32 head vectors form a point cloud under Hellinger distance. Its finite
zero-dimensional persistent-homology death times are exactly the edge weights
of a minimum spanning tree. The code preserves the full 31 death times for
selected detail samples and reports layer-resolved summaries for every token:

- mean and maximum death time;
- persistence entropy;
- largest persistence gap;
- centered spectral effective rank;
- head-level local intrinsic dimension.

This is not regression. It is a multiscale description of how quickly head
routing coalitions merge as the neighborhood radius grows.

### H3 — Prompt detection without integration

Some heads may still retrieve prompt evidence while other heads route through
response states that themselves have little prompt ancestry.

Prompt provenance is composed through ordered layers. Prompt states are fixed
anchors with provenance one; response states start at zero. For head `h`, layer
`l`, and response token `t`, the known routing contribution is

\[
q_{l,h,t}^{\mathrm{known}}
 = m^{P}_{l,h,t}
 + a^{\mathrm{self}}_{l,h,t} g_{l,t}
 + \sum_{j<t} a_{l,h,t,j} g_{l,j}.
\]

Because censored mass has unknown endpoints, the audit keeps bounds:

\[
q^{\mathrm{lower}}=q^{\mathrm{known}},
\qquad
q^{\mathrm{upper}}=q^{\mathrm{known}}+m^{\mathrm{unresolved}}.
\]

The next-layer routing proxy is the mean across heads. This equal-head average
is deliberately described as an attention-only proxy because `W_O V` is not
available. The saved layer fields include mean grounding, head polarization,
bound width, and supported/unsupported response feedback.

### H4 — Fracture-to-lock-in dynamics

Hallucination need not remain permanently disordered. The predicted trajectory
is:

```text
head-routing reorganization
    -> prompt ancestry decreases
    -> routing concentrates on fewer response anchors
    -> anchor turnover and temporal route velocity decrease
```

The audit therefore distinguishes onset from later occupancy. It measures exact
response-source effective number, top-1 share, recent-history mass, mean lag,
anchor turnover, and permutation-invariant layer/token transitions. Head sets
are compared with sliced Wasserstein distances over fixed random projections;
head index is never assumed to retain the same semantic role across layers.

### H5 — Exact endpoint topology matters

A topology claim requires more than layer/head marginals. The score stage builds
an endpoint-rewired null that preserves:

- layer, head, target token, and edge weight;
- prompt-versus-response role;
- prompt-position bin or response log2-lag bin;
- row edge count and row retained mass;
- exact self-attention.

It resamples only the exact source endpoint. If real and rewired results are
indistinguishable, the evidence supports routing marginals rather than source
ancestry/topology. The current null does not preserve exact source in-degree;
that stronger fixed-margin null belongs in a later confirmatory experiment.

## Architecture

```text
config.py
    feature registry, mechanism families, pre-registered phase directions

routing.py
    ResearchSample -> retained edges -> role simplex + exact RR endpoints
    constrained exact-endpoint rewiring

geometry.py
    Hellinger head point clouds, H0 persistence, effective rank, LID,
    permutation-invariant layer/token transitions

provenance.py
    ordered-layer prompt-provenance lower/upper bounds

features.py
    combines fracture, integration, concentration, and dynamics fields

reference.py
    unlabeled task/causal-position robust reference and family atypicality

experiment.py
    fit and score stages; writes frozen label-free artifacts

evaluation.py
    label-only post-hoc AUROC/AUPRC, onset effects, late-span lock-in,
    and real-versus-rewired comparisons

main.py / run.sh
    command-line and one-command runner
```

## Artifact granularity

The main score artifact preserves

```text
[token, layer, mechanism feature]
```

rather than reducing the response to a few sample-level statistics. Selected
samples can additionally save

```text
[token, layer, head, role]
[token, layer, H0 death time]
[token, layer, head, provenance bound]
retained exact edge endpoints and weights
```

by passing `--detail-sample-id` or setting `DETAIL_SAMPLE_IDS` in `run.sh`.

## Run

Smoke test:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_LIMIT=20 TEST_LIMIT=5 DETAIL_SAMPLE_IDS=12471 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/attention_phenomenology/run.sh
```

Full audit:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/attention_phenomenology/run.sh
```

Outputs:

```text
experiments/attention_phenomenology/outputs/full/
  reference.npz
  scores/
    manifest.json
    samples/<sample_id>.npz
    details/<selected_sample_id>.npz
  evaluation/
    evaluation.json
    family_metrics.csv
    layer_feature_metrics.csv
    onset_layer_effects.csv
    lockin_layer_effects.csv
    onset_phase_curves.csv
```

## Interpretation rules

1. A high persistent-homology summary means head routing is fragmented; it does
   not mean semantic evidence is contradictory.
2. Low prompt-provenance lower bounds can reflect censoring. The corresponding
   upper bound and unresolved-mass control must be inspected jointly.
3. Low temporal velocity and high source concentration indicate route lock-in,
   not factual error by themselves.
4. The strongest support for an attention topology mechanism requires an onset
   pattern, a later lock-in pattern, and a drop under endpoint rewiring.
5. This project deliberately does not add Dirichlet spectra, reconstruction
   residuals, a GNN, or a learned fusion score before the proposed mechanisms
   have been empirically validated.
