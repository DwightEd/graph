# HoloRoute: Neural Learning on Dual-Axis Causal Attention Event Graphs

## Research objective

HoloRoute is an attention-only, token-level, unsupervised hallucination detector.
It is motivated by one mechanism hypothesis:

> Hidden-state reorganization changes later query-key routing. Normal computation
> therefore induces reusable composition laws between Transformer depth and
> autoregressive token relay, while hallucination may leave a position-independent
> structural residual in this attention event geometry.

The model does not reconstruct hidden states, MLP outputs, value vectors, or
causal logit contributions. Its claim is limited to a learned attention-routing
footprint.

## 1. Input graph

HoloRoute reuses the audited `AttentionEventGraph` from
`experiments/attention_holonomy_audit/graph.py`.

For each retained token pair and layer, one event node stores the complete head
profile

\[
x_{s,t,l}=[A^{l,1}_{t,s},\ldots,A^{l,H}_{t,s}]\in\mathbb R^H,
\]

with a separate observation mask for cache-censored heads. The graph contains:

- depth edges \(v_{s,t,l}\to v_{s,t,l+1}\);
- depth-respecting relay edges \(v_{u,s,l}\to v_{s,t,l+1}\);
- query sets containing events that enter the same target and layer;
- causal diamonds that admit relay-then-depth and depth-then-relay compositions.

No prompt/response ratio, near/far state machine, Markov state, CUSUM, or
handcrafted lock-in score is used by the neural method.

## 2. Head-profile event encoder

Each observed head is encoded from its attention value, observation bit, head
identity, layer, source role, lag, source position and target position. A small
Transformer encodes interactions among the heads of one event, followed by
learned attention pooling. This keeps head identity and head coalitions instead
of flattening or averaging them before graph learning.

## 3. Relation-specific low-rank transport

Messages are not added in a shared coordinate system. For relation type
\(r\in\{\text{depth},\text{prompt-relay},\text{response-relay}\}\), HoloRoute
uses

\[
T_r(z_s,z_t)z_s
=z_s+U_r\left(g_r(z_s,z_t)\odot V_r^\top z_s\right),
\]

where \(U_r,V_r\) are low-rank relation parameters and \(g_r\) is an
event-conditioned gate. This is a lightweight typed-transport design inspired
by the principle of transporting heterogeneous messages before aggregation.

## 4. Causal path and query-set aggregation

Transported relay predecessors are combined with target-conditioned segmented
attention rather than an unweighted sum. Query coalitions are encoded with
inducing-point set attention over all retained incoming events for one
`(target, layer)` group.

A message block therefore has four components:

\[
z_e,\quad m_e^{depth},\quad m_e^{relay},\quad m_e^{query}.
\]

A learned gate fuses the four components, followed by a residual feed-forward
update. Multiple blocks propagate higher-order path and query-set context.

## 5. Holonomy residual

For an audited causal diamond, HoloRoute composes its learned transports along
both valid routes:

\[
z_A=T_D T_R z,\qquad z_B=T_R T_D z.
\]

Normal computation need not have zero curvature. An auxiliary predictor learns
the expected route difference from the start and end event states. The
holonomy residual is

\[
E_{hol}=\|z_A-z_B-\mu_\theta(z_{start},z_{end})\|_2^2.
\]

It is an auxiliary objective and a local diagnostic, not a forced primary
hallucination score.

## 6. Label-free objectives

Training never reads hallucination labels.

### Whole-event masking

The complete \(H\)-head vector of selected events is hidden. The graph encoder
must reconstruct it from depth, causal-path and query-set context.

### Relay-path completion

Selected relay relations are removed only when another predecessor remains.
The remaining causal-path context predicts the successor event profile.

### Depth and query completion

Depth context predicts the next-layer event profile. Query-set context predicts
one event from the other source events entering the same query/layer group.

### Holonomy and anti-collapse

The model minimizes expected holonomy residual and variance/covariance
regularization on event states.

The full training loss is

\[
\mathcal L=\mathcal L_{event}+\lambda_P\mathcal L_{path}
+\lambda_D\mathcal L_{depth}+\lambda_Q\mathcal L_{query}
+\lambda_H\mathcal L_{hol}+\lambda_V\mathcal L_{var/cov}.
\]

## 7. Local mechanism vector and anomaly score

No prefix accumulation is used. Multiple deterministic masking rounds estimate
six local token quantities:

1. whole-event reconstruction error;
2. path prediction error;
3. depth prediction error;
4. query-set prediction error;
5. depth/relay context disagreement;
6. causal-diamond holonomy residual.

A disjoint unlabeled calibration stream regresses these values on nuisance
variables including absolute and relative position, response length, graph
counts, retained mass, observed-head coverage, unresolved mass and task. A
robust covariance model is fitted to the standardized residual vector. The final
score is the empirical upper-tail energy

\[
S_t=-\log_{10}\widehat P(E_{train}\ge E_t).
\]

Position is therefore a conditioning variable, not a score accumulator.

## 8. Required ablations

The implementation provides frozen configurations for:

- `event_only`;
- `no_path`;
- `no_depth`;
- `no_query_set`;
- `identity_transport`;
- the full model.

A graph or mechanism claim is valid only when the full model improves both
unlabeled completion and label-posthoc hallucination detection over the matched
ablation.
