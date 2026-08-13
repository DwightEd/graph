# Compact token graph representation

## Scientific contract

The primary node vector is the complete windowed layer-head Lookback tensor. For a 32-layer, 32-head observer it has 1024 coordinates. No layer/head average is used before saving, train-only calibration, PCA projection, or anomaly scoring.

Test labels are loaded only after node vectors, compact graph state, anomaly scores, sample selection, and graph indices have been frozen. The sparse cache is censored at `attention_floor`; absent entries are never presented as recovered original attention.

## Direct node state

For response token $t$, layer $l$, head $h$, prompt length $P$, retained prompt mass $p$, retained history mass $r$, and saved diagonal $d$:

\[
L_{t,l,h}=\frac{p/P}{p/P+(r+d)/(t+1)}.
\]

A causal temporal window preserves every layer-head coordinate. The flattened tensor is the only primary node representation; auxiliary graph mechanisms are not concatenated before validation.

## Sparse layer routes

Propagating every auxiliary mechanism through all layer-head channels is redundant. Instead, identical `(query, source)` edges within one layer are aggregated using the mean of the strongest $K$ head values:

\[
B_{l,t,s}=K^{-1}\sum_{h\in\operatorname{TopK}}a_{l,h,t,s}.
\]

Unobserved head values are zero, the divisor remains $K$, and routes below `attention_floor` are removed. This differs from an all-head mean: it preserves strong minority-head evidence while requiring either sufficient strength or cross-head support.

The direct compact state records retained prompt mass, prompt coverage/span/centroid/centroid change, and history mass/coverage/edge fraction/lag/lag change/far-history fraction per layer. History edge fraction is retained explicitly because it was the strongest earlier scalar structural signal; it is no longer hidden inside an indiscriminate feature average.

## Multi-hop prompt provenance

Direct prompt evidence is represented by raw position moments

\[
S^{(0)}_{t,l}=\sum_{p<P}B_{l,t,p}[1,u_p,u_p^2],
\]

and propagated through the layer-level RR route:

\[
S^{(k)}_{t,l}=\sum_{r<t}B_{l,t,r}S^{(k-1)}_{r,l}.
\]

No row normalization occurs. The three propagated moments recover path-weighted prompt mass, centroid, and spread, so a weak path cannot become equivalent to a strong path. Two hops explicitly represent non-adjacent prompt inheritance through an earlier response token.

## Unsupervised reference and evaluation boundary

An unlabeled train reservoir fits position-conditioned median/MAD calibration and a PCA subspace on the 1024-D Lookback vectors. The deployable label-free scores are robust tail deviation and PCA reconstruction error. PCA supplies population/sample coordinates but is not optimized with test labels.

After artifacts freeze, every Lookback layer-head and every compact mechanism layer is evaluated independently. Raw AUROC preserves direction; `max(AUC,1-AUC)` is post-hoc association only. Any selected layer, head, or mechanism must be frozen on validation before confirmatory held-out testing.

## Sample visualization

The static visualization never places prompt and response tokens on two horizontal lines. It uses four complementary coordinate systems: the frozen 1024-D Lookback PCA with weighted RR edges, a prompt-to-response weighted adjacency matrix, a causal response-to-response adjacency matrix, and an RR target-versus-lag plot. Matrix color and graph edge width/color encode salient route weight. Distance from the RR diagonal and the lag ordinate encode exact token distance. Hop-1 inherited prompt centroid and spread are overlaid on the RP matrix. Every response token is rendered in representation space; hallucination labels are used only for coloring and target-row annotation.
