# Causal evidence-flow graph representation

## Scientific contract

The direct node vector is the complete windowed layer-head Lookback tensor. For a 32-layer, 32-head observer it has 1024 coordinates. Graph detection uses a separate fixed multiscale filter bank; no message parameter is trained and no label is read before all representations, controls, scores, thresholds, and sample choices freeze.

Test labels are loaded only after node vectors, compact graph state, anomaly scores, sample selection, and graph indices have been frozen. The sparse cache is censored at `attention_floor`; absent entries are never presented as recovered original attention.

## Direct node state

For response token $t$, layer $l$, head $h$, prompt length $P$, retained prompt mass $p$, retained history mass $r$, and saved diagonal $d$:

\[
L_{t,l,h}=\frac{p/P}{p/P+(r+d)/(t+1)}.
\]

A causal temporal window preserves every layer-head coordinate. The flattened tensor is the only primary node representation; auxiliary graph mechanisms are not concatenated before validation.

## Sparse layer routes

The topology is the union of retained edges across heads within each layer. Identical `(query, source)` edges receive their strongest retained head value:

\[
B_{l,t,s}=\max_h a_{l,h,t,s}.
\]

Routes below `attention_floor` are removed. No head average is used, so a strong minority-head edge is preserved rather than diluted.

The direct compact state records retained prompt mass, prompt coverage/span/centroid/centroid change, and history mass/coverage/edge fraction/lag/lag change/far-history fraction per layer. History edge fraction is retained explicitly because it was the strongest earlier scalar structural signal; it is no longer hidden inside an indiscriminate feature average.

## Multiscale evidence flow

Let $W_l$ be the raw, non-normalized RR route at layer $l$. Conditional messages and their raw path reliabilities are

\[
M_l^{(1)}=(W_lX_l)/(W_l\mathbf1),\quad
M_l^{(2)}=(W_l^2X_l)/(W_l^2\mathbf1).
\]

The node filter bank retains $(W\mathbf1)(X-M^{(1)})$ and $(W^2\mathbf1)(M^{(1)}-M^{(2)})$. This raw-mass gate is continuous at zero, so an arbitrarily weak route cannot carry a full conditional residual. Prompt sources are represented by 16-bin routing distributions rather than only three moments, and the same mass-gated wavelet differences are retained for these distributions. $W\mathbf1$ and $W^2\mathbf1$ remain separate reliability channels.

## Structural null and evaluation

For every graph, a null topology preserves each RR target, layer, route-entry count, and individual edge weight, but replaces the source with a seeded uniformly sampled causal predecessor; parallel null routes are allowed. Both true and randomized fields receive independent train-only one-class calibration. The primary structural claim is accepted only when true propagation improves over this null and the complete evidence-flow detector improves over token-only in both AUROC and AUPRC. Uncertainty is a paired sample-level bootstrap, preserving all tokens belonging to one response.

## Unsupervised reference and evaluation boundary

Aligned unlabeled train reservoirs fit position-conditioned median/MAD and PCA references for token-only, direct-edge, true-propagation, and randomized-propagation fields. Each view combines robust top-tail deviation and PCA residual through train empirical CDFs. The fixed train quantile produces node anomalies; RR connectivity among those nodes defines anomaly components.

After artifacts freeze, every Lookback layer-head and every compact mechanism layer is evaluated independently. Raw AUROC preserves direction; `max(AUC,1-AUC)` is post-hoc association only. Any selected layer, head, or mechanism must be frozen on validation before confirmatory held-out testing.

## Sample visualization

The static visualization uses the true-topology propagation embedding with weighted RR edges, RP and RR weighted adjacency matrices, and a score/component panel. Node fill color is the label-free anomaly score. Hallucination labels appear only as outlines or marks after evaluation and never influence coordinates, scores, thresholds, or component construction.
