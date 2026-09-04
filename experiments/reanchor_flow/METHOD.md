# Method: claim-boundary re-anchor flow

## 1. Pre-registered observations

The experiment tests three observations, not a learned feature combination.

1. During ordinary continuation, evidence/prompt incoming mass should decline
   while response-history mass rises.
2. At a factual claim boundary, correct generation should exhibit an evidence
   reread pulse and an evidence-seeded global path through the first claim
   tokens to the claim sink.
3. Hallucinated claims should more often lack this re-anchor flow. The graph
   interpretation is accepted only if it survives attention-only, direct-edge,
   edge-bag, middle-layer, and role/lag-preserving rewiring controls.

## 2. Causal coordinates

For response token `x[p]`, the predictive query is `q=p-1`. A query-row message
is lifted into the event edge `s -> p`, never `s -> q`. Thus the token just
generated at `p` can become a source for later predicted tokens, while no fake
self-loop is introduced at the query coordinate.

## 3. Edge capacity from the same frozen model

For query head `h` and its actual GQA KV head `g(h)`, define

\[
\kappa_{l,h,q,s}
=A_{l,h,q,s}\|W_l^{O,[h]}V_{l,g(h),s}\|_2.
\]

The main graph averages this capacity over all heads and layers. This matches
the endpoint-level path cuts, which remove the selected source-target pair in
all layers and heads. The same forward also produces two controls: an all-layer
attention graph and a middle-third functional graph. Target columns are
normalized over causal sources to form `W`.

The capacity records residual-write magnitude, not support or veto. Signed
causal meaning is supplied only by the real path-deletion reruns.

## 4. Target-conditioned global flow

For claim sink `t`, backward path potential is

\[
h_t(i)=\sum_{k=i+1}^{t}W_{i,k}h_t(k),\qquad h_t(t)=1.
\]

It equals the sum of products over all directed paths from `i` to `t`. For the
boundary set `B` containing the first claim tokens, define

\[
g_t(i)=
\begin{cases}
h_t(i),&i\in B,\\
\sum_{k>i}W_{i,k}g_t(k),&i\notin B.
\end{cases}
\]

Then

\[
F_E(B\to t)=|E|^{-1}\sum_{s\in E}g_t(s)
\]

is the evidence-to-sink path mass that first crosses the claim boundary set.
The closure fraction divides this by
`|E|^{-1} sum_s h_t(s)`. A response-seeded version is reported separately; it
is not treated as additive provenance because response nodes can themselves
carry evidence.

## 5. Claim proxy and local observations

The pilot uses sentence-like spans defined by decoded punctuation and newlines.
The first `anchor_width` tokens form `B`; the last token is the claim sink.
This boundary is label-free and route-independent, but not a semantic claim
annotation. Stronger controlled experiments should replace it with fixed
atomic claims and support/validator spans.

For every response token, the graph also reports incoming evidence, other
prompt, and response-history shares. The claim-start reread pulse subtracts the
median evidence inflow over the preceding fixed window from evidence inflow at
the boundary.

## 6. Structural controls

- **Attention:** identical DAG and flow algorithm, edge weights from `A` only.
- **Middle functional:** the same functional capacity using the middle third of
  layers, matching the layer choice in FlowTracer-style analyses.
- **Direct:** one-hop evidence mass entering the claim sink.
- **Bag:** average evidence edge mass across claim tokens, without path
  incidence.
- **Rewire:** within each target, permute edge weights among sources with the
  same evidence/prompt/response role and logarithmic lag bin.

All are evaluated on the same claims. A graph gain is meaningful only when its
source-cluster bootstrap interval is positive.

## 7. Causal path audit

For a claim sink, the Doob-conditioned transition is

\[
P_t(i,k)=\frac{W_{i,k}h_t(k)}{h_t(i)}.
\]

Uniform evidence-source flow is propagated through `P_t`; the highest edge
flows covering a fixed mass define a functional backbone. Its token endpoint
pairs are removed in the original model at `query=p-1`, post-softmax and
pre-Value-sum. The same model then recomputes every downstream residual,
attention, and MLP operation. Attention-backbone, top-capacity edge-bag, and
role/lag-matched endpoint masks are rerun identically.

Graph structure is causally supported only when

\[
|\Delta\mu_{functional}|>
|\Delta\mu_{attention}|,
|\Delta\mu_{bag}|,
|\Delta\mu_{matched}|.
\]

No hallucination label participates in selecting audited sources, claims, or
paths.

## 8. Scope and stopping rules

The experiment can establish a recurring global structural correlate and test
whether paths selected by the graph matter to the observer model. It does not
yet distinguish support facts from validator constraints, isolate MLP
parametric knowledge, or recover a different generator's original free
trajectory.

Stop or narrow the claim when any of the following occurs:

- correct claim boundaries do not show a reproducible evidence reread pulse
  over mid-claim controls;
- functional global flow does not beat direct, bag, attention, and rewired
  controls;
- role/lag rewiring preserves the result;
- the selected functional backbone is no more influential than matched cuts;
- the effect exists only in one task and fails to transfer.
