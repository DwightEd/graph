# Method

## 1. Anchor-resolved lineage

Let `a` index a prompt anchor and let relay depth be `k in {0, 1, 2+}`. The state

\[
G_{t,l,h}(a,k)
\]

is the attention-routing mass for response token `t`, layer `l`, and head `h` that can be traced to anchor `a` through `k` response endpoints. A separate response-base source and unresolved sink are maintained.

Direct prompt edges add mass to depth zero. A message through a response endpoint shifts depth

\[
0\to1,\qquad1\to2+,\qquad2+\to2+.
\]

Self attention carries the previous layer's state. Sparse-cache remainder is assigned to the unresolved sink. Each row is mass-conserving.

## 2. De Bruijn-style event graph

A layer event is

\[
e=(s\to t,l,x_e),\qquad x_e\in\mathbb R^H.
\]

The high-order relation

\[
(u\to s,l-1)\to(s\to t,l)
\]

preserves the middle response token and the order of the causal walk. Order-2 and order-3 contexts are weighted means and standard deviations of predecessor event head vectors. They are not flattened source-count features.

## 3. Nested predictive gate

The order-1 feature contains current layer head-resolved role routing. Order 2 adds direct-anchor, one-hop lineage, and two-walk contexts. Order 3 adds multi-hop lineage and three-walk contexts.

Each model predicts the next layer's role routing and anchor-lineage state with fixed-alpha ridge regression. The audit reports

\[
\Delta_2=L_1-L_2,\qquad \Delta_3=L_2-L_3.
\]

A matched-dimension null shuffles only the newly added block before fitting and scoring. Therefore

\[
\Delta^{path}_2=L^{null}_2-L_2
\]

measures path correspondence rather than width alone.

## 4. Anchor congruence and recoupling

Direct and relay anchor distributions are normalized only when both have sufficient observed mass. Their Jensen-Shannon divergence is

\[
D_{t,l}=D_{JS}(p^{direct}_{t,l},p^{relay}_{t,l}).
\]

A train-only reference supplies high and low thresholds. Recoupling depth is the number of subsequent layers required for a high-divergence state to fall below the low threshold.

## 5. Audit escape and lock-in

A token is response-local when response-base mass is high and anchor-connected mass is low relative to train-only thresholds. For a future horizon `K`:

\[
P_R(t)=\frac1K\sum_{j=1}^K 1[Z_{t+j}=R],
\]

\[
P_E(t)=1[\exists j\le K: Z_{t+j}=E].
\]

The lock-in proxy is

\[
S_t=\tilde D_t\,P_R(t)\,[1-P_E(t)].
\]

It does not mark every large transition as hallucination. It requires anchor disagreement, persistent response-local routing, and absent evidence escape.

## 6. Evaluation alignment

The cached state after response token `t` is aligned to the label of token `t+1`:

```text
score[:-1] <-> labels[1:]
```

This avoids using the erroneous token's own post-token query state to claim pre-token detection.
