# Method and implementation plan

## 1. Relationship to the token graph project

The projects operate at different levels:

- the token graph project studies adjacency structure and graph-conditioned
  node representations;
- this project studies how evidence routes and hidden states evolve over layer
  depth and autoregressive time.

They can later be connected by treating a graph-produced node vector as one
trajectory signal, but neither project imports the other. Attention-route
dynamics and Hessian-inspired geometry therefore remain independently
testable, and a failure here cannot be hidden by a graph feature mixture.

## 2. Input constraints

Only existing compressed attention and hidden-state arrays are eligible. Raw
text, token embeddings, hallucination labels, and supervised feature selection
are excluded. Every reference transform is fit on the official train split and
frozen before evaluation labels are opened.

The implemented attention reader accepts the formal sparse response-CSR fields:

```text
response_idx
token_ids
attention_diagonal       [layers, heads, tokens]
response_row_ptr         [layers * heads * response_tokens + 1]
response_column_indices
response_values
attention_floor
```

Rows are ordered by `(layer, head, response_query)`. Sparse values contain only
earlier-token attention above the extraction floor. The self-attention diagonal
is stored separately.

The hidden-state stage will use a separate sidecar with the canonical contract:

```text
hidden_states            [layers + 1, tokens, hidden_dim]
post_attention_states    [layers, tokens, hidden_dim]  optional
post_ffn_states           [layers, tokens, hidden_dim]  optional
```

Only `hidden_states` is necessary for layer trajectories. The optional sublayer
states are required before making separate claims about attention routing and
FFN overwrite.

## 3. Stage A: route-dynamics encoding (implemented)

For layer `l`, head `h`, and response target `t`, retained attention is mapped
to a fixed anchor vector

\[
q_{lht}=[q^{prompt}_{1:B},q^{history}_{1:K},q^{self},q^{unresolved}].
\]

Prompt anchors are fixed relative-position bins. Response-history anchors are
causal lag bins `(1, 2, 4, 8, 16, 32, farther)`. The unresolved category is

\[
q^{unresolved}_{lht}=\max(1-a_{tt}-\sum_{s<t}a_{ts},0).
\]

Retained edges are never renormalized. A normalized copy including the explicit
unresolved category is used only where a probability distribution is required
by Jensen-Shannon divergence.

For every layer/head route, the implementation calculates:

\[
d^{time}_{lht}=JS(q_{lht},q_{lh,t-1}),
\]

\[
d^{depth}_{lht}=JS(q_{lht},q_{l-1,h,t}),
\]

\[
a_{lht}=\|q_{lht}-2q_{lh,t-1}+q_{lh,t-2}\|_2,
\]

and generalized head disagreement

\[
d^{head}_{lt}=H(\bar q_{lt})-\frac1H\sum_h H(q_{lht}).
\]

The full `(layer, head, anchor)` state, temporal difference, temporal
acceleration, and depth difference are mapped to a fixed-dimensional token
vector by deterministic CountSketch. This preserves channel-specific changes
without fitting weights and without averaging heads or layers.

### Gate A

Before adding hidden states, verify that route embeddings and diagnostics are:

1. invariant to file batching and processing order;
2. sensitive to source rewiring while preserving target/head/layer marginals;
3. more informative than direct prompt mass on at least one held-out task;
4. not explained entirely by response position or response length.

Failure at Gate A means route dynamics remains a diagnostic, not a detector.

## 4. Stage B: graph-conditioned hidden trajectory encoding (implemented)

Apply one shared train-frozen projection `R` to every layer:

\[
X_l=\operatorname{scale}_l(H_lR).
\]

Do not fit an independent PCA per layer because its rotations are not aligned.
Compute attention and FFN updates when the sublayer states exist:

\[
u^{attn}_{lt}=h^{attn}_{lt}-h^{pre}_{lt},\qquad
u^{ffn}_{lt}=h^{post}_{lt}-h^{attn}_{lt}.
\]

The primary graph-related residual is a closed-form, train-only prediction of
the next state update from the actual attention neighbors:

\[
M_{lh}=A_{lh}X_l,
\]

\[
\widehat{\Delta X_l}=[X_l,M_{l1},\ldots,M_{lH},u_l]B_l,
\]

\[
E_{lt}=\Delta X_{lt}-\widehat{\Delta X}_{lt}.
\]

`B_l` is solved by robust trimmed ridge regression. This is label-free matrix
estimation, not a GNN and not backpropagation. To keep the closed-form problem
bounded without averaging heads, a fixed balanced signed hash maps heads to a
small number of message channels. The mapping is frozen by the run seed and
saved in the model; every head contributes to exactly one channel. Layer
trajectories are retained with fixed orthonormal DCT coefficients rather than a
layer mean.

The implemented node control contains the projected target state, response
position, retained prompt share, length-normalized off-diagonal Lookback,
self mass, and unresolved mass. Both total routing allocation and candidate-
token-normalized RP/RR preference are therefore controlled before source
identity is credited to the graph.

The fit/calibration split is made by complete training sample. Three copies of
the same state equation are fit independently: node control, true graph, and a
causal source-rewired graph. Rewiring preserves layer, head, target, edge
weight, RP/RR type, prompt position bin, and response-lag bucket.

### Gate B

Compare the same frozen unsupervised detector on:

```text
hidden-only
hidden + true attention neighbors
hidden + causally rewired neighbors
true layer order
shuffled layer order
```

The graph-conditioned block advances only if true topology improves over both
hidden-only and rewired topology under sample-level paired bootstrap.

The current implementation performs the first, label-free part of this gate by
reporting sample-held-out state-prediction MSE. AUROC/AUPRC are deliberately
absent from representation construction and belong to the later frozen
detector evaluation.

## 5. Stage C: Hessian-inspired local geometry

The ICML Hessian Geometry paper reconstructs a Fisher metric from a learned
log-partition function. That exact estimator requires neural-network training
and is not copied here. We use its phase-boundary insight with a nonparametric,
train-frozen local metric.

For a token representation `z`, estimate from train neighbors:

\[
G(z)=(\operatorname{Cov}(N_k(z))+\lambda I)^{-1}.
\]

For consecutive response tokens, compute metric speed

\[
s_t=\sqrt{(z_t-z_{t-1})^TG(z_{t-1})(z_t-z_{t-1})},
\]

trajectory turning/curvature, and the local metric-change score

\[
b_t=\frac{\|G(z_t)-G(z_{t-1})\|_F}
{\|z_t-z_{t-1}\|_2+\epsilon}.
\]

These quantities test whether a token crosses an unusual representation phase
boundary. They are reported separately from route-dynamics features.

### Gate C

The geometry block must beat Euclidean speed, global Mahalanobis distance, and
position-conditioned hidden-only references. Otherwise Hessian-inspired
geometry is rejected rather than added to a joint score.

## 6. Frozen evaluation protocol

No block is averaged into a score before its individual ablation is complete.
Fit all scalers, projections, neighborhood references, and thresholds on train.
Evaluate only after representations are frozen. Report:

- token AUROC and AUPRC;
- fixed train-quantile precision, recall, and F1;
- hallucination-span/component IoU;
- task- and position-stratified results;
- sample-level paired bootstrap for every claimed gain.

A route/graph claim requires

\[
\text{true topology}>\text{hidden only}
\quad\text{and}\quad
\text{true topology}>\text{rewired topology}.
\]

Two-dimensional UMAP or diffusion-map plots are explanatory artifacts only and
are never used as evidence that a representation detects hallucination.
