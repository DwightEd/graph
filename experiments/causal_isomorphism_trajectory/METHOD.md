# Causal Isomorphism Trajectory Geometry

## Status

This directory develops **Causal Isomorphism Trajectory Geometry (CITG)**, an
attention-only, label-free method for token-level hallucination detection.

CITG replaces the neural CMRP detector. CMRP established that exact RR source
identity can be learned beyond a coarse lag baseline, but its routing-surprise
score was not aligned with hallucination labels. The new method therefore does
not predict the next source and does not equate rarity, unpredictability, or
large motion with factual error.

The scientific question is narrower and falsifiable:

> Does a token follow a common, task- and causal-phase-conditioned trajectory of
> multiplex attention topology across generation time and model depth, or does
> its graph state/transition occupy a low-density region?

The score is fitted without correctness or hallucination labels. Labels are
opened only after the score artifact is frozen.

---

## 1. Evidence motivating the method

The repository currently supports three observations.

1. Retained-attention entropy decreases around hallucination onset in the
   paired analysis. This suggests concentration or premature routing collapse
   can be informative.
2. A historical RR spectral residual obtained moderate token-level separation.
   The useful signal was distributed across multiple channels rather than
   dominated by one head.
3. CMRP's exact-source topology gate could pass while its full hallucination
   detector remained near random. Thus source topology can be learnable without
   source-prediction loss being a correctness signal.

`LLM Reasoning as Trajectories: Step-Specific Representation Geometry and
Correctness Signals` analyzes hidden states as ordered trajectories and finds
that correctness information is especially strong in late trajectory states
and transitions. Its final predictor is supervised, but its structural lesson
transfers: align functional stage, retain state and transition, and do not
assume that errors simply move farther or more chaotically.

Temporal-WL and consistent-event-graph work provides the graph counterpart:
static snapshots can miss time-respecting paths, while temporal invariants can
represent how causal events compose. CITG uses a bounded two-hop event-graph
signature inspired by those results. It is not a complete temporal-isomorphism
test.

Primary references:

- Sun et al., *LLM Reasoning as Trajectories*, ACL 2026.
- Souza et al., *Provably Expressive Temporal Graph Networks*, NeurIPS 2022.
- Gao and Ribeiro, *On the Equivalence Between Temporal and Static Equivariant
  Graph Representations*, ICML 2022.
- Heeg et al., *Isomorphisms in Temporal Graphs: A Category-Theoretic
  Perspective*, 2025.

---

## 2. Data and causal graph contract

All attention access goes through:

```python
from research_dataset import open_research_dataset
```

and:

```python
sample.iter_sparse_attention_blocks(...)
```

No experiment module parses NPZ/PT caches directly.

For response token `t`, a retained event is:

```text
(source, target=t, layer, head, retained_weight, relation, causal_lag)
```

with:

```text
RP: prompt source   -> response target
RR: response source -> response target
```

The information direction is:

```text
attended source -> attending response token
```

A missing legal CSR entry means only:

```text
attention <= attention_floor
```

not exact zero. CITG models retained events and never treats all missing edges
as observed negative edges.

Prompt token identities are not compared across samples. Prompt events carry
only the invariant RP role, exact layer/head and retained weight. Relative
prompt centroids are not used.

---

## 3. Band-balanced multiplex events

A 32-layer, 32-head model has 1024 attention channels. Global top-k selection
would favor a few high-mass layers and erase depth evolution. CITG partitions
model depth into `B` ordered layer bands and, for every target token and band,
retains a deterministic top-k set:

```text
up to K_RP RP events per band
up to K_RR RR events per band
```

Selection uses retained weight with canonical CSR order as the tie break.

Before selection, the extractor records full retained summaries overall and by
layer band:

```text
RP/RR mass
RP/RR edge count
sum(w log w)
maximum retained weight
```

Thus the hashed topology uses a bounded salient event graph while scalar role
statistics still use every retained event.

---

## 4. Rooted temporal-isomorphism signature

### 4.1 Edge labels

For selected event `e`, define the discrete label:

\[
q(e)=
(\mathrm{relation},
 \mathrm{layer\ band},
 \mathrm{head},
 \mathrm{lag\ bin},
 \mathrm{weight\ bin}).
\]

RR lag is binned by:

\[
b_{\mathrm{lag}}(t-j)
=
\min(\lfloor\log_2(t-j)\rfloor,B_{\mathrm{lag}}-1).
\]

Weight is binned relative to `attention_floor`, so the label does not assume
uncensored dense attention.

### 4.2 One-hop rooted events

For current token `t`, CITG counts labels of all selected incoming events:

\[
\Phi_t^{(1)}[q]
=
\#\{e:u\rightarrow t,\ q(e)=q\}.
\]

A second histogram accumulates log-scaled retained weight under the same keys.

### 4.3 Time-respecting two-hop paths

For each selected RR event:

\[
j\rightarrow t,
\]

and selected incoming event:

\[
u\rightarrow j,
\]

CITG counts the ordered label pair:

\[
\Phi_t^{(2)}[q(u\rightarrow j),q(j\rightarrow t)].
\]

This represents prompt/RR ancestry and response-history composition without
requiring token identities to be comparable across samples.

### 4.4 Source-sharing motifs

For every RR source used by the current token, the signature also records:

- multiplicity of selected edges from that source;
- causal-age bin;
- selected parent-event count;
- number of layer bands using the source.

This distinguishes multiple independent sources from repeated multiplex use of
one response anchor.

### 4.5 Fixed-width hashing

One-hop events, two-hop paths and source motifs are deterministically hashed
into count and weight histograms. Each half is L1-normalized.

The signature is invariant to renaming response nodes when causal order, roles
and edge labels are preserved. It is intentionally incomplete:

- finite hashing can collide;
- two-hop refinement cannot distinguish every non-isomorphic graph;
- exact sub-floor edges are unavailable;
- prompt semantics are not represented.

Accordingly, the method is described as **temporal-isomorphism-inspired** or a
bounded temporal-WL invariant, not as a complete isomorphism solver.

---

## 5. Two trajectory axes

### 5.1 Generation-time state and transition

Let the global rooted signature and invariant role features form token state:

\[
x_t=[r_t\Vert\Phi_t].
\]

CITG retains both state and causal transition:

\[
\Delta_t x=x_t-x_{t-1}.
\]

It also records cosine/L1 signature motion, role-state motion, source-anchor
turnover, route effective rank and cross-channel consensus.

This allows both kinds of failure:

```text
excessive movement / unstable routing
premature collapse / abnormally small movement
```

to become low-density events. No score direction is chosen from test labels.

### 5.2 Model-depth trajectory

Head indices are not assumed to have the same semantics across adjacent layers.
Within each ordered layer band, CITG builds a head-labelled event signature,
then compares adjacent band signatures:

\[
d_{t,b}^{\mathrm{depth}}
=
D(\Phi_{t,b-1},\Phi_{t,b}),
\]

where `D` includes cosine and L1 distance.

The representation includes:

- all adjacent-band distances;
- mean depth movement;
- late-band mean and maximum movement;
- early-to-late slope;
- depth curvature;
- token-to-token change in the depth trajectory.

This is the attention-graph analogue of retaining late hidden-state trajectory
transitions rather than only a final static state.

---

## 6. Semantically invariant scalar state

The method includes only cross-sample quantities with common roles:

```text
RP/RR retained mass
RP/RR edge fraction
retained-weight entropy
RR retained-weight entropy
top-1 retained share
effective number of selected RR sources
RR source top-1 share
source lag mean/std
active channel fraction
route effective rank
cross-channel route consensus
anchor turnover
```

It excludes:

```text
prompt_centroid
prompt_provenance_centroid_hop1
prompt_provenance_spread_hop1
```

because prompt-relative positions are not semantically aligned across samples.

---

## 7. Preregistered representation variants

Four variants are fitted before labels open:

### `full`

\[
[\text{state},\text{generation transition},
  \text{depth trajectory},\text{depth transition}].
\]

This is the primary method.

### `static`

Current role/topology state plus current depth trajectory, without token-time
differences.

### `topology`

Hashed event topology and its time/depth transitions, excluding role-mass
features.

### `mass`

Role/concentration state and its time transition, excluding hashed topology.

These frozen variants answer whether any gain comes from:

- static state;
- temporal dynamics;
- graph topology;
- or only coarse mass/concentration statistics.

The primary method is never selected after inspecting test labels.

---

## 8. Phase-conditioned unlabeled geometry

CITG does not train a neural classifier. Complete source groups from the
unlabeled train split are partitioned into:

```text
fit groups
calibration groups
```

For each token, the condition is:

```text
task_type × causal log2 position bucket
```

The position bucket uses only `t+1`, never final response length.

Within sufficiently populated conditions, each feature is standardized using
train-only median and MAD, with a global fallback. A robust PCA/PPCA model is
then fitted to sampled fit tokens.

For standardized trajectory vector `z`, PPCA energy is:

\[
E(z)=
\frac{1}{D}
\left[
\sum_{i=1}^{d}\frac{a_i^2}{\lambda_i}
+
\frac{\|z-\hat z\|_2^2}{\sigma^2}
\right].
\]

Unlike orthogonal residual alone, this score detects:

- off-subspace escape;
- extreme coordinates inside the common subspace;
- unusually large transitions;
- unusually small/premature transitions, if they deviate from the conditioned
  reference trajectory.

The disjoint calibration stream converts energy to a monotone finite-sample
upper-tail score:

\[
S_t=-\log\widehat P_{\mathrm{cal}}(E\ge E_t).
\]

No correctness label, head selection, score inversion or test-selected fusion
appears in fitting or calibration.

---

## 9. Required topology gate

For calibration tokens, every RR event with a legal alternative is rewired to
another source in the same coarse lag bin. The control preserves:

```text
target
layer
head
relation
retained weight
coarse lag bin
```

and changes exact source identity.

CITG recomputes the complete trajectory and defines:

\[
\Delta_{\mathrm{rewire}}
=
E_{\mathrm{rewired}}-E_{\mathrm{true}}.
\]

Gaps are averaged within source group and source-group bootstrapping produces a
95% interval. The gate passes only when:

- rewire coverage exceeds the preregistered minimum; and
- the lower confidence bound is positive.

A failed gate means the hashed invariant has not demonstrated useful
source-sensitive topology, even if a post-hoc AUROC happens to be high.

---

## 10. Frozen evaluation protocol

```text
unlabeled fit source groups
    -> condition scalers and PPCA geometry

unlabeled calibration source groups
    -> empirical score tails and topology gate

held-out test attention
    -> frozen full/static/topology/mass token scores

frozen score artifact
    -> labels open only for AUROC/AUPRC
```

Score artifacts bind:

- train and test dataset manifests;
- fitted reference digest;
- fit/calibration/test source groups;
- complete response token rows;
- online-causal temporal scope.

---

## 11. Falsifiable success criteria

CITG is not accepted because code runs. A useful result requires:

1. **Topology gate:** rewired trajectories have reliably higher calibration
   energy than true trajectories.
2. **Held-out detection:** frozen `full` score beats `mass` and current spectral
   baselines on identical rows.
3. **Dynamic necessity:** `full` beats `static`.
4. **Topology necessity:** `full` or `topology` beats `mass`.
5. **Depth necessity:** removing layer-band transitions degrades performance.
6. **No direction selection:** inverted test AUROC is never used as a method.
7. **Leakage audit:** fit/calibration/test source groups and labels remain
   isolated.

If these gates fail, the method is recorded as another negative result rather
than repaired by test-label score selection.

---

## 12. Claim boundary

CITG can test whether hallucination correlates with low-density attention graph
state/transition geometry. It cannot establish:

- that attention is factual evidence;
- that common unlabeled trajectories are necessarily correct;
- that its hashed signature is a complete graph invariant;
- that prompt tokens are semantically aligned;
- that sub-floor edges are absent;
- that routing convergence equals confidence.

Semantic evidence flow remains a separate future extension requiring an
explicit prompt-role segmentation contract.
