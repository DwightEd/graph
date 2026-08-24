# Attention Holonomy Mechanism Audit

## Research question

This audit tests one narrow attention-only mechanism before a neural detector is
implemented:

> Hidden-state reorganization may leave a footprint in later query-key routing.
> If normal computation has a reusable local transport law, the same attention
> event should be explainable both from its previous-layer continuation and from
> causal routes entering its source token. Hallucination may be associated with
> abnormal disagreement between these two structural explanations.

The audit does **not** claim to recover MLP, residual, value, or logit
contributions. It studies an attention-routing proxy and keeps hallucination
labels sealed during fitting and scoring.

## 1. Dual-axis causal attention event graph

For every retained token-pair event in layer `l`, all heads are grouped into one
node

\[
v_{s,t,l}=\big[A^{l,1}_{t,s},\ldots,A^{l,H}_{t,s}\big].
\]

The observation mask is stored separately because a missing channel is censored
below the attention floor, not observed zero.

The graph contains three structural objects:

1. **Depth edges**
   \[
   v_{s,t,l}\rightarrow v_{s,t,l+1}.
   \]
2. **Depth-respecting relay edges**
   \[
   v_{u,s,l}\rightarrow v_{s,t,l+1}.
   \]
3. **Query sets** containing all events entering the same target and layer.

A causal diamond exists when

\[
(u\rightarrow s,l),\;(u\rightarrow s,l+1),\;
(s\rightarrow t,l+1),\;(s\rightarrow t,l+2)
\]

are all observed. It allows two valid compositions from the first event to the
last: relay-then-depth and depth-then-relay.

## 2. Compositional head geometry

For an existing event, censored heads receive one fixed sub-floor value and the
head vector is normalized to a simplex distribution `p`. Transport operators are
fitted in centered-log-ratio coordinates

\[
\operatorname{clr}(p)_h=\log p_h-\frac1H\sum_j\log p_j.
\]

This avoids treating a compositional head profile as an unconstrained Euclidean
vector. Errors are measured with squared Hellinger distance after mapping back
to the simplex.

## 3. Train-only structural transports

The unlabeled transport-fit source groups estimate small affine ridge maps:

- `D_l`: same token pair from layer `l` to `l+1`;
- `R_l^P`: a prompt-origin event entering a response source and continuing to a
  later target;
- `R_l^R`: the analogous response-origin relay;
- a local event predictor using only role, lag, position, mass, and observation
  coverage;
- a query-set predictor that additionally uses the mean and dispersion of the
  other incoming events.

These maps are measurement instruments, not the final HoloRoute neural model.
Their purpose is to establish whether the proposed relations have predictive
content before a trainable transport network is justified.

## 4. Frozen mechanism coordinates

For each response token, the audit reports six pre-registered high-risk
coordinates:

1. `depth_transport_error`: failure of the same pair's depth continuation;
2. `relay_transport_error`: failure of causal predecessor paths to explain the
   successor event;
3. `relay_path_dispersion`: disagreement among multiple causal predecessor
   predictions;
4. `depth_relay_disagreement`: disagreement between depth and relay predictions
   of the same event;
5. `query_set_error`: leave-one-event-out error using the remaining source set;
6. `diamond_holonomy`: non-commutativity of relay/depth transport around a
   causal diamond.

For a diamond starting at `z`,

\[
\Omega=H^2\!\left(D_{l+1}R_lz,\;R_{l+1}D_lz\right),
\]

where `H^2` is squared Hellinger distance.

The audit also freezes four structural controls:

- depth transport gain over a layer-conditioned mean;
- relay transport gain over a typed layer-conditioned mean;
- query-set gain over a metadata-only predictor;
- real relay gain over a matched middle-node rewire.

A complex neural module is authorized only when its corresponding unlabeled
control is positive on held-out source groups.

## 5. Position-conditioned reference

The previous rupture detector correlated almost perfectly with token position.
This audit never uses a prefix sum or CUSUM. Each token is scored locally.

On disjoint calibration groups, each mechanism coordinate is regressed on
nuisance variables:

- relative position and polynomial terms;
- response length;
- event, relay, and diamond counts;
- retained mass;
- observed-head fraction;
- unresolved mass;
- task identity.

The frozen score is a robust standardized residual. The exploratory joint score
is the mean positive squared residual across available mechanism coordinates.
Position is therefore a conditioning variable, not evidence of hallucination.

## 6. Claim boundary

Passing the audit can establish that attention event geometry contains a
position-independent structural footprint. It cannot establish that MLP or
residual computation caused the footprint. A later neural method and base-model
intervention are required for stronger mechanism claims.
