# Causal Typed-Path De Bruijn Routing

## Research question

This experiment asks whether a hallucination is preceded by a change in the
*grammar of attention lineage* and followed by a persistent response-closed
routing regime.  The method is attention-only, token-level, label-free during
fit and scoring, and prefix-causal.  It does not train a GNN, a probe, an NRI
encoder, or any other model by back-propagation.

The method has three deliberately narrow components:

```text
mass-conserving layer-unfolded lineage
    -> per-(layer, head) order-2 De Bruijn grammar
    -> recent rupture x route-state lock-in
```

The path-only detector is evaluated on its own.  A previously frozen causal RR
residual may later be combined through the explicit bridge, but it must not
hide a failed path mechanism.

## 1. Observable graph and claim boundary

All attention is read through `research_dataset.open_research_dataset`.  The
cache contains response-query rows only.  An off-diagonal event is

```text
(source, target, layer, head, retained_weight)
```

with information direction `source -> target`.  Prompt-query rows, values
below `attention_floor`, value/output projections, residual branches, and MLP
updates are unavailable.  Consequently, this experiment constructs an
**attention-lineage proxy**, not the complete Transformer causal graph.
The row for token `t` is observed after that token has been processed. Thus the
detector is online/prefix-causal for detecting the current token, not a
pre-emission predictor of token `t`.

The current dataset exposes only one prompt/response boundary.  It does not
expose trustworthy evidence/question/system spans, so the implementation never
fabricates `evidence`, `question`, or `supporting fact` edge types.  Those types
may be added only through a future validated central data view.

For every response token `t`, layer `l`, and head `h`, retained off-diagonal
mass is separated into prompt and earlier-response mass.  The exact diagonal
is the self mass.  If their sum is `m`, a correction is applied only when
float16 storage makes `m > 1` by at most the frozen numerical tolerance
`1e-3`:

\[
\widetilde a_{lhtj}=a_{lhtj}/\max(1,m_{lht}),\qquad
u_{lht}=1-\sum_j\widetilde a_{lhtj}.
\]

`u` is an unresolved sink.  A missing CSR edge is therefore never interpreted
as an observed zero, and a floor-filled dense matrix is never used as a Markov
operator. A larger overshoot is treated as a duplicate/corrupt attention row
and fails closed; it is never normalized into apparently valid data.

## 2. Layer-unfolded typed-path automaton

The primary representation is a mutually exclusive five-state distribution:

```text
P0       direct prompt lineage at the current layer
P_PLUS   prompt lineage transmitted through at least one response relay
R0       response-token base lineage propagated only by self
R_PLUS   response-closed lineage that traversed a response-to-response edge
U        unresolved lineage
```

Before the first attention layer, every response token is in `R0`.  Let
`q[l,h,t]` be its state after one head at layer `l`.  Because the cache does not
contain `W_O`, the transport from the previous layer is a fixed,
permutation-invariant head mean, stated as an assumption rather than learned:

\[
\bar q_{l-1,t}=H^{-1}\sum_{h'}q_{l-1,h',t}.
\]

For the current row, prompt edges enter `P0`; the diagonal keeps the previous
state; an RR edge maps `P0/P_PLUS` to `P_PLUS`, maps `R0/R_PLUS` to `R_PLUS`,
and leaves `U` unresolved.  The remaining mass enters `U`.  Thus every
`[token, layer, head]` state is non-negative and sums to one.  No layer/head
channel is averaged in the stored output.

`typed_path_dp.py` additionally implements a same-channel, finite-horizon token
path decomposition with near/far ordered RR motifs.  It is an ablation and an
explanation tool.  It is explicitly not presented as a physical same-head path
through Transformer depth.

## 3. Soft order-2 De Bruijn grammar

Generation time is a second axis, distinct from Transformer depth.  For channel
`c=(l,h)`, the frozen normal grammar is fitted from complete unlabeled source
groups:

\[
N_{c,a,b,d}=\sum_s\sum_{t\ge2}
q_{s,t-2,c,a}q_{s,t-1,c,b}q_{s,t,c,d},
\]

\[
\Theta_c(d\mid a,b)=
\frac{N_{c,a,b,d}+\alpha}
     {\sum_{d'}N_{c,a,b,d'}+\alpha M}.
\]

The implementation keeps the most probable fixed number of soft states per
token and renormalizes them before accumulating expected counts.  This is a
bounded sparse-soft approximation; it is deterministic and recorded in the
reference.  Order two is fixed before labels are opened.  Order one and a
within-sample time-shuffle are controls, not label-selected alternatives. The
time shuffle is explicitly an offline, non-causal sequence null: it can move a
later row earlier inside a bucket and must never be reported as a token/onset
detector.

## 4. Rupture and lock-in

The De Bruijn model predicts the next route distribution for every channel.
For soft observed state `q` and predicted state `p`, transition surprise is the
cross-entropy

\[
H(q,p)=-\sum_d q_d\log p_d,
\]

which matches the fractional-count likelihood used to fit the grammar. The
implementation also saves bounded Jensen--Shannon departure. A causal decaying
memory retains a recent rupture rather than requiring every later token to
remain surprising.

Lock-in is large only when `R_PLUS` persists, the grammar predicts continued
detached routing, and consecutive route states are stable.  The primary raw
channel statistic is

\[
s_{t,c}=\operatorname{recent\_rupture}_{t,c}
        \operatorname{lockin}_{t,c}.
\]

Prompt-lineage mass is stored as a diagnostic, and its causal positive drop is
used to fit a recorded robust reference. Its primary-score weight defaults to
zero: earlier onset audits found that prompt mass does not have one universal
hallucination direction.

The word *lock-in* refers to persistence in the finite De Bruijn route-state
graph.  The causal token DAG itself has no literal cycle or stationary
attractor.

## 5. Label-free fitting and calibration

Complete `source_id` groups are deterministically divided into three disjoint
streams:

```text
fit groups                  De Bruijn counts and robust phase scales
channel-calibration groups  one upper-tail ECDF per layer/head
fusion-calibration groups   dependence-aware calibration after channel fusion
```

The 1024 channel scores are converted to upper-tail p-values only after every
channel has been computed.  They are combined with a symmetric Cauchy
statistic, then calibrated again on the third stream.  This preserves channel
identity through the complete structural computation while avoiding a
label-fitted channel weight or best-head selection.

The core fit entry rejects any manifest whose split is not `train`, and the
core score entry rejects any manifest whose split is not `test`; CLI option
names alone are not trusted. Fit and score processes open data with embedded
labels sealed. Evaluation
captures the frozen score artifact, verifies its digest, manifest, complete
token rows, source-group audit, and reference provenance, and only then opens
labels in a separate process.

## 6. Required controls

The endpoint null keeps target, layer, head, edge weight, RP/RR role, causal
validity, near/far type, and coarse lag bin while changing legal response
sources.  It preserves the one-hop role masses exactly and perturbs only
lineage.  A useful graph mechanism should assign the rewired calibration paths
more anomaly than the true paths.

The pre-registered comparisons are:

1. five-state typed lineage versus one-hop role state;
2. exact endpoints versus the endpoint-rewired null;
3. order two versus order one and an offline within-sample time shuffle;
4. rupture-times-lock-in versus each factor alone;
5. path-only versus the frozen causal RR residual;
6. the fixed hybrid versus RR residual alone;
7. QA, Data2txt, and Summary reported separately with source-cluster
   uncertainty.

If exact paths do not beat the one-hop and rewired controls, the method cannot
claim a topology contribution.  If order two does not beat order one, the
De Bruijn component should be removed rather than defended post hoc.

## 7. Relationship to prior methods

PathNN contributes the idea that ordered paths, rather than an unordered
neighborhood mean, are the object to preserve.  De Bruijn graph modeling
contributes an explicit higher-order temporal state.  Neural Sheaf, NRI, and
NIIP motivate typed transport, relation separation, and additive intervention
tests, but their learned modules are intentionally excluded from the core:
they would reintroduce back-propagation and make it unclear which mechanism
produced a gain.  A fixed or unlabeled Procrustes transport can be studied only
after the identity/equal-transport baseline passes.
