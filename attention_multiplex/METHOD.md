# Attention Dynamic Multiplex

## Scope

This module studies the structure of cached Transformer attention. It does not
train a hallucination detector, read token labels, compute AUROC, aggregate
hidden states, or use a GNN.

For one sample, the available tensor is

\[
\mathcal A_{l,h,t,s},
\]

where `l` is ordered depth, `h` is a relation type, `t` is a response query,
and `s` is a prompt or causal response source. The cache contains exact
self-attention diagonals and off-diagonal response rows whose weights are at
least `attention_floor`.

## Data boundary and reconstruction

All access goes through `research_dataset.py`. No file in this subproject may
call `torch.load` or `numpy.load` on attention caches.

The central data class exposes one channel as a censored causal matrix
`[R,N]`, not a false square `[N,N]` matrix:

- retained off-diagonal edges: exact cached values;
- diagonal: exact `attention_diagonal` values;
- legal unretained edges: imputed with `attention_floor` (normally `0.01`);
- future/non-causal entries: structural zero;
- PP query rows: unavailable and not fabricated.

Every dense view includes `observed`, `eligible`, and `censored` masks. The
channel iterator prevents allocation of `[L,H,R,N]`.

## Dynamic multiplex graph

The graph has response query roles and prompt/response source roles. Layer is
ordered depth, and head is an edge relation. It is represented by the
unfolding

\[
B_{(l,t),(h,s)}=\mathcal A_{l,h,t,s}.
\]

This does not average heads or layers and does not symmetrize the directed
query/source relation.

The floor-filled matrix itself is dense and contains a large deterministic
causal baseline. The spectral input removes only that baseline:

\[
B^{mass}_{(l,t),(h,s)}=
\begin{cases}
A_{l,h,t,s}-\tau,&A_{l,h,t,s}\text{ retained off diagonal},\\
A_{l,h,t,t},&s=t\text{ and the diagonal is included},\\
0,&\text{censored or ineligible},
\end{cases}
\]

where \(\tau=\texttt{attention\_floor}\). A second, separate view uses
`sqrt(A)-sqrt(tau)` to represent probability-distribution shape. These two
views are not concatenated or assigned arbitrary weights.

For each view, a joint truncated SVD

\[
B\approx U\Sigma V^T
\]

produces

\[
Z^{query}_{l,t}=U_{(l,t)}\Sigma^{1/2},\qquad
Z^{source}_{h,s}=V_{(h,s)}\Sigma^{1/2}.
\]

The output retains `query_by_layer[L,R,d]` and
`source_by_head[H,N,d]`. It does not average them into a scalar or a single
token vector.

## Claim boundary

The coordinates are jointly aligned across layer/head channels inside one
sample. They are not yet aligned across independent samples. This phase is a
construction and attention-structure study, not evidence of hallucination
separation.
