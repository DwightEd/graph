# Causal Multiplex Routing Prediction

## Status

This directory develops a new attention-only, label-free method for token-level
hallucination detection.  The working name is **Causal Multiplex Routing
Prediction (CMRP)**.

The first implementation is deliberately narrower than the eventual evidence-
flow model.  It models exact response-history source identities, causal order,
layer/head channels, retained edge weights, and prompt-versus-response roles.
It does **not** interpret relative prompt positions as semantic evidence and it
does not claim that attention is factual support.

The central scientific question is:

> Does a generated token use a source-aware, multi-layer/multi-head causal
> routing pattern that is predictable from normal unlabeled generation, or does
> it enter an unusual response-history routing regime?

The primary score is learned without hallucination labels.  Labels are opened
only after the score artifact has been frozen.

---

## 1. Why this project is needed

### 1.1 What the existing experiments established

The repository currently contains evidence that the response-to-response (RR)
causal attention prefix carries a useful unsupervised signal.  The strongest
historical RR spectral result is the reconstruction energy outside a dominant
cross-layer/head subspace.  This suggests that attention organization matters,
but the current triangular-Laplacian representation is still close to a
source-usage summary and does not preserve complete source-target paths.

The previous causal-topology baseline also showed that simply appending a menu
of one-hop/two-hop statistics and applying a one-class detector is not enough to
establish a useful topology gain.  The next model must therefore learn a
self-supervised structural task rather than merely concatenate more hand-built
statistics.

### 1.2 What CHARM contributes and what it leaves unresolved

CHARM constructs one union token graph from all Transformer layer/head
attention matrices.  Each union edge carries the full layer-head attention
fingerprint, and an edge MLP mixes those channels before graph message passing.
This demonstrates a practical way to retain multiple attention matrices.
However:

1. it is supervised with hallucination labels;
2. layer/head channels are mixed into one edge embedding;
3. a two-hop union path may switch layer/head channels between hops;
4. its gain can come from the high-dimensional edge MLP rather than topology;
5. it does not learn the normal causal graph-generation process.

CMRP keeps the exact source-target identity and the layer/head identity of every
retained event, then learns a label-free source-prediction task.  It explicitly
reports a lag-preserving source-rewiring control so that a useful model must do
better on the true topology than on a counterfactual with similar marginal
statistics.

### 1.3 Why prompt-position centroids are retired

Earlier exploratory code compressed prompt attention into relative-position
moments such as `prompt_centroid` and propagated those moments over RR edges.
That construction is mathematically valid but not semantically aligned across
samples: position 0.8 may represent a question, evidence, a demonstration, or a
table field in different prompts.

Therefore CMRP does not consume:

```text
prompt_centroid
prompt_provenance_centroid_hop1
prompt_provenance_spread_hop1
```

Prompt edges are initially represented only by the invariant role
`prompt -> response`, their layer/head channel, and their retained weight.
Semantic evidence roles are a future extension that requires an explicit,
validated prompt segmentation contract.

---

## 2. Data and graph contract

Every attention access goes through:

```python
from research_dataset import open_research_dataset
```

and then through:

```python
sample.iter_sparse_attention_blocks(...)
```

Experiment code must not parse NPZ/PT caches directly.

For response token `t`, a retained attention event is

```text
(source, target=t, layer, head, weight, relation)
```

where relation is one of:

```text
RP: prompt source   -> response target
RR: response source -> response target
```

The information direction used by the graph is:

```text
attended source -> attending response token
```

For a 32-layer, 32-head model, each edge belongs to one of 1024 channels.  The
method does not average those channels before encoding.

### 2.1 Censoring boundary

The formal cache stores exact diagonal values and retained off-diagonal values
at or above `attention_floor`.  A missing legal edge means only:

```text
attention <= attention_floor
```

not exact zero.  Phase 1 models the topology of **retained events only**.  It
never treats every missing edge as a negative example.  A future censored-
likelihood extension may model the missing legal support explicitly.

### 2.2 Dynamic graph view

A response creates a causal graph trajectory:

```text
G_0 -> G_1 -> ... -> G_T
```

At token `t`, only prompt sources and response sources `< t` are legal.  Node
states are updated sequentially, so an RR source state can carry multi-hop
history into later tokens.

---

## 3. Event selection without changing the data semantics

A long response may contain many retained events across 1024 channels.  CMRP
uses deterministic typed top-k selection per target token for computational
control:

```text
up to K_RP prompt events
up to K_RR response-history events
```

Selection is by retained weight with deterministic tie breaking.  Full retained
RP/RR mass and count are still recorded before selection and supplied as role-
level summaries.

This means the neural encoder studies a reproducible salient-event graph while
also knowing the full retained role marginals.  The method must report the
selection values and should later ablate them.

---

## 4. Channel-preserving source-aware graph encoder

Let `c=(l,h)` denote a layer/head channel.  Every selected event receives:

- source node state;
- retained attention weight;
- RP/RR relation embedding;
- layer embedding;
- head embedding;
- causal-lag Fourier features for RR edges.

For an RR event `j -> t`:

\[
 m_{j\to t}^{c}
 = \phi_\theta\!\left(
    h_j,
    \log(1+w_{j\to t}^{c}/\tau),
    e_l,
    e_h,
    e_{RR},
    \gamma(t-j)
 \right).
\]

For an RP event, the source state is a learned generic prompt-anchor state and
lag features are zero:

\[
 m_{p\to t}^{c}
 = \phi_\theta\!\left(
    h_P,
    \log(1+w_{p\to t}^{c}/\tau),
    e_l,
    e_h,
    e_{RP},
    0
 \right).
\]

This does not claim that all prompt tokens are semantically identical.  It is a
role-level Phase-1 representation chosen because prompt source identities are
not aligned across samples.

For token `t`, messages are pooled by both mean and maximum, then concatenated
with full retained RP/RR mass and count summaries:

\[
 a_t = \rho_\theta\!\left(
   \operatorname{mean}_{e\in E_t}m_e,
   \operatorname{max}_{e\in E_t}m_e,
   \log(1+M_t^{RP}),
   \log(1+M_t^{RR}),
   \log(1+N_t^{RP}),
   \log(1+N_t^{RR}),
   \operatorname{position}(t)
 \right).
\]

The dynamic token state is

\[
 h_t = \operatorname{GRUCell}(a_t,h_{t-1}).
\]

Because an RR message contains `h_j`, later states can depend on exact earlier
source identities and multi-hop routing.  Layer/head identity is retained in
every event before the shared message network mixes information.

---

## 5. Label-free self-supervision

### 5.1 Retained-RR presence prediction

Before consuming token `t`'s incoming events, the previous dynamic state predicts
whether at least one retained RR edge exists:

\[
 \widehat p_t^{RR}
 = \sigma(g_{presence}(h_{t-1},t)).
\]

The target is derived from the attention cache, not from a hallucination label.
The binary negative log-likelihood is

\[
 \mathcal L_t^{presence}
 = \operatorname{BCE}(\widehat p_t^{RR},\mathbf 1[N_t^{RR}>0]).
\]

### 5.2 Source-identity contrastive prediction

For each selected observed RR edge `j -> t` in channel `c`, the model predicts
which prior response node was selected.

The query is based only on the state before token `t` and the edge channel:

\[
 q_{t,c}=g_q(h_{t-1},e_l,e_h,t).
\]

A candidate prior source `k<t` is represented by

\[
 s_{k,t}=g_s(h_k,\gamma(t-k)).
\]

and receives score

\[
 z(k\mid t,c)=q_{t,c}^{\top}s_{k,t}/\sqrt d.
\]

The candidate set contains:

1. the true source;
2. a deterministic lag-bin-preserving rewired source when one exists;
3. additional deterministic negatives, prioritizing the same lag bin.

The source loss is InfoNCE/cross-entropy:

\[
 \mathcal L_{t,e}^{source}
 = -\log
 \frac{\exp z(j\mid t,c)}
 {\sum_{k\in\mathcal C_{t,e}}\exp z(k\mid t,c)}.
\]

This is the raw negative log-probability.  It is never divided by
`log(candidate_count)`: candidate-set size is part of the fixed contrastive
task, not a post-hoc loss normalization.

Using same-lag hard negatives reduces the shortcut of selecting a source only
from its distance to the target.

### 5.3 Edge-weight diagnostic

Given the true source representation, a small head predicts the retained edge
log-weight.  Smooth-L1 error is used as an auxiliary training term and is saved
as a diagnostic.  It is not the primary anomaly score.

### 5.4 Training objective

The first implementation trains with

\[
 \mathcal L
 = \operatorname{mean}_t
   \left[
     \mathcal L_t^{presence}
     + \mathbf 1[N_t^{RR}>0]
       \operatorname{mean}_{e\in E_t^{RR}}\mathcal L_{t,e}^{source}
   \right]
 + \lambda_w\mathcal L^{weight}.
\]

No correctness or hallucination label appears in this objective.

---

## 6. Primary anomaly score

For each token, the raw routing surprise is

\[
 r_t
 = \mathcal L_t^{presence}
 + \mathbf 1[N_t^{RR}>0]
   \operatorname{mean}_{e\in E_t^{RR}}
   \mathcal L_{t,e}^{source}.
\]

Complete source groups in the unlabeled training split are deterministically
partitioned into:

```text
fit groups          -> optimize the neural model
calibration groups  -> freeze the empirical distribution of r_t
```

The calibrated score is the global finite-sample upper-tail transform:

\[
 S_t
 = -\log
   \widehat P_{cal}(r\ge r_t).
\]

The transform is monotone and does not use token labels or position-specific
post-hoc direction selection.

Saved diagnostics include:

```text
raw_route_surprise
presence_nll
source_nll
weight_error
rewired_source_nll
rewire_gap = rewired_source_nll - source_nll
rewire_edge_gap = L_source(rewired edge) - L_source(true edge)
```

---

## 7. Required topology gate

A source-aware model must assign lower surprise to the true source than to a
lag-bin-preserving counterfactual source on held-out unlabeled calibration
groups:

\[
 \Delta_{rewire}
 = \mathbb E[
   \mathcal L^{source}_{rewired}
   - \mathcal L^{source}_{true}
 ] > 0.
\]

The reference artifact reports:

- mean and median rewire gap;
- fraction of evaluated edges with a positive gap, as a diagnostic only;
- evaluated-edge count, selected-edge count, and their coverage;
- a pass flag that is true exactly when at least one finite rewired edge was
  evaluated and the preregistered mean edge gap is positive.

The gate aggregates `rewire_edge_gap` over evaluated edges, never a mean of
per-token means.  A selected RR edge without a legal lag-bin-preserving rewire
is retained in the selected count but not the evaluated count.

Failure of this gate means the model has not demonstrated source-sensitive
topology learning, even if a later benchmark AUROC is above chance.

Future controls must add:

- channel shuffling while preserving each channel marginal;
- response-time shuffling;
- a mass-only encoder;
- a CHARM-style union-edge MLP without message passing;
- channel-consistent versus cross-channel multi-hop paths.

---

## 8. Evaluation protocol

The execution contract is:

```text
unlabeled train fit groups
    -> neural parameters                     labels never opened

unlabeled train calibration groups
    -> score calibration + topology gate     labels never opened

unlabeled test attention
    -> source-overlap audit + frozen scores  labels never opened

frozen score artifact
    -> AUROC/AUPRC and conditioned benchmark labels opened here only
```

The test score artifact contains `sample_id`, `token_index`, metadata, frozen
fit/calibration/test source-group audit fields, and `score*` fields so that
`experiments/conditioned_benchmark/` can align CMRP with RR spectral and other
frozen methods on identical token rows. Evaluation rechecks that artifact's
digest and only accepts the canonical `test` split before labels unlock.

Primary comparisons:

1. CMRP primary score versus RR spectral primary score;
2. true topology versus lag-preserving rewired topology gate;
3. source prediction versus retained RP/RR mass-only baselines;
4. token-level and response-level conditioned evaluation by task;
5. hallucination-onset profile after the model is frozen.

---

## 9. What Phase 1 can and cannot claim

### It can test

- whether exact RR source identity is predictable from earlier graph states;
- whether layer/head-aware routing has information beyond coarse lag;
- whether hallucination tokens have high held-out routing surprise;
- whether true topology is preferred to lag-preserving rewiring;
- whether a dynamic source-aware model improves over RR spectral summaries.

### It cannot yet claim

- that prompt attention is factual evidence;
- that prompt token identity is semantically comparable across samples;
- that missing attention edges are exact negatives;
- that the contrastive source score is a normalized likelihood over every legal
  edge;
- that every multi-hop path remains in one fixed layer/head channel;
- that routing concentration equals confidence or correctness.

---

## 10. Planned Phase-2 extensions

### 10.1 Semantic prompt roles

Introduce a validated prompt segmentation contract such as:

```text
instruction | demonstration | context | evidence | question | schema
```

Prompt nodes can then receive role embeddings and exact within-sample source
identity.  Only after this step should the method use the term evidence flow.

### 10.2 Censored edge likelihood

Model retained edges with a continuous likelihood and missing legal edges with
censoring probability:

\[
 -\log p_\theta(w_e),\qquad e\in E_{observed},
\]

\[
 -\log P_\theta(W_e\le\tau),\qquad e\notin E_{observed}.
\]

### 10.3 Spectral consistency

Use the existing RR causal spectrum as an auxiliary reconstruction constraint,
not as a test-selected score fusion:

\[
 \mathcal L_{spectral}
 = \|\lambda_K(G_t)-\lambda_K(\widehat G_t)\|_2^2.
\]

### 10.4 Temporal isomorphism controls

Add order-preserving temporal-WL or event-graph signatures and compare them with
mass-, lag-, time-, and channel-preserving rewires.

---

## 11. Falsifiable success criteria

The method is not accepted merely because the code trains.  A useful result
requires all of the following:

1. **Topology gate:** calibration rewire gap is positive with stable coverage.
2. **No shortcut:** true-source performance exceeds a lag-only candidate model.
3. **Held-out detection:** frozen test AUROC/AUPRC exceed the mass-only and RR
   spectral baselines under identical conditioned rows.
4. **Dynamic necessity:** removing previous node states degrades source
   prediction and detection.
5. **Channel necessity:** removing layer/head identity degrades performance.
6. **Leakage audit:** fit, calibration, test, and labels remain separated by
   source group and execution stage.

A failed gate is reported as a failed hypothesis rather than hidden by selecting
another diagnostic after test labels are observed.
