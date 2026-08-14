# Exact-channel Lookback graph representation

## Node state and graph route

For every response token $t$, the direct state is the complete layer-head
Lookback tensor flattened without averaging:

\[
X_t=\operatorname{vec}\left(L_{t,l,h}\right)\in\mathbb{R}^{32\times32}
=\mathbb{R}^{1024}.
\]

When a Lookback denominator is structurally undefined, its coordinate is
filled with the cache `attention_floor` (normally `0.01`). The sparse
attention cache remains sparse: an absent edge is not turned into an edge of
weight `0.01`.

Each retained attention entry is an exact directed route
`(layer, head, source, target, weight)`. Its channel is
`layer * heads + head`; no heads are unioned, maximized, averaged, or randomly
projected.

## One-hop evidence flow

For each channel $c=(l,h)$, a response node receives two same-channel,
one-hop messages:

\[
F^P_{t,c}=\sum_{s\in\mathrm{prompt}}a_{t,s,c}(X_{t,c}-0.01),\qquad
F^R_{t,c}=\sum_{s<t,\,s\in\mathrm{response}}a_{t,s,c}(X_{t,c}-X_{s,c}).
\]

The primary graph node vector is therefore

\[
Z_t=[X_t,F^P_t,F^R_t]\in\mathbb{R}^{3072}.
\]

It makes the question testable: does exact prompt-to-response or
response-to-response routing add anomaly-relevant information beyond the
1024-D token state? This is fixed message passing, not a GNN: no backpropagation
or learned edge weights are used.

## Views and structural control

The validation compares seven frozen views: `scalar_only`, `token_only`,
`prompt_graph`, `response_graph`, `true_graph`, `rewired_graph`, and
`direct_marginals`. `true_graph` is $Z$; `rewired_graph` preserves every RR
target, channel, edge weight, and RP/RR type, but replaces an RR source with a
causal source from the same $\lfloor\log_2(\mathrm{lag})\rfloor$ bin. Thus it
tests source identity rather than merely mass, degree, or lag scale.

An unsupervised scorer is fitted from unlabeled train vectors only. Test labels
are read only after node vectors and scores are frozen, for diagnostic AUROC/
AUPRC reporting. The primary comparisons are `true_graph` versus
`token_only`, `true_graph` versus `rewired_graph`, and `response_graph` versus
`prompt_graph`.

The corrected scalar Lookback-ratio check is a separate supervised post-hoc
diagnostic: train labels choose its direction and fit its nuisance-adjusted
probe. It is not the `scalar_only` unsupervised graph baseline and is executed
only after the graph artifacts have frozen.
