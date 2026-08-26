# Original-code DBGNN reference

This experiment asks one narrow question: does an order-2 causal-walk lift add
useful information to the final token representation beyond a first-order GCN?
It is a reference baseline, not the proposed hallucination mechanism.

## Graph mapping

The input is a saved `EncodedTokenGraph`. Its old `node_embedding` is ignored;
only token identity, prompt/response boundary and retained typed attention edges
are read.

For a retained edge `u -> v` at Transformer layer `l` and head `h`, define the
head-averaged temporal event

\[
e_l(u,v)=\frac{1}{H}\sum_h a^{l,h}_{v,u}.
\]

The first-order graph keeps token nodes and aggregates an endpoint pair across
layers:

\[
w_1(u,v)=\frac{1}{L}\sum_l e_l(u,v).
\]

An order-2 node is an endpoint pair `(u,v)`. A high-order edge exists only for
a layer-respecting relay:

\[
(u,v)\rightarrow(v,w),\qquad
1\le l_2-l_1\le\delta.
\]

Its weight is the sum of `e_l1(u,v) * e_l2(v,w)` over matching temporal events.
The default `delta=1` therefore means that information arriving at `v` in one
Transformer layer may be relayed from `v` in the next layer. Token causality is
already guaranteed by the saved edges (`source < target`).

The original paper uses graph-specific identity matrices. Those cannot share a
model across variable-size prompt/response graphs. The only input adaptation is
a fixed feature contract:

```text
x_fo[token]      prompt/response role + causal bounded position
x_ho[(u,v)]      x_fo[u] || x_fo[v] || pair weight || causal bounded lag
```

No GroundedRoute embedding, hallucination label, hand-built anomaly statistic,
lineage tensor, degree, entropy or residual is an encoder input.
The bounded position and lag transforms depend only on the current indices;
they never divide by the eventual response length.

## Original neural operator

`vendor/dbgnn.py` and `vendor/gcn.py` contain the core model files from
`lisiq/dbgnn` commit `2613afe5c63183229470164f5decc2bca1a1826e`.

The copied `HO_GCN` performs:

```text
order-2 path graph -- two weighted GCN layers --+
                                                 +-- terminal-node projection
first-order graph -- two weighted GCN layers ----+        -> token state
```

The paper code applies a linear classifier after this tensor. The adapter
replaces only that final `mlp` with `Identity`, so the exported value is
`node_embedding[token_count, embedding_dim]`. The `gcn` option calls the copied
first-order model and exports the same-sized tensor.

## Label-free fit and node-only detection

The original repository trains with node labels and cross entropy. This
experiment instead holds out endpoint pairs, removes them from both the
first-order graph and every derived order-2 path, and trains a shared bilinear
decoder against role/lag-matched unretained endpoints. Because the sparse cache
censors values below its floor, these are not claimed to be exact zero edges.
This objective learns the encoder;
its reconstruction error is not the anomaly score.

After training, the clean complete graph is encoded once. The downstream
detector receives only the frozen response-node matrix:

```text
node_embedding -> robust PCA whitening -> kNN distance -> token score
```

Fit, validation and detector-calibration source groups are disjoint. Only the
third group is written to `calibration/index.npz`; test sources are disjoint
from all three.

The copied terminal projection gives prompt nodes no output state because the
cache contains no attention rows targeting prompt tokens. Endpoint training
therefore uses only response-to-response pairs whose source already has an
incoming path and whose target retains another incoming pair after holdout.
Prompt edges still participate in the first/high-order message paths, but this
reference does not claim to learn prompt-endpoint identity directly. Solving
that limitation requires a new residual/typed grounding architecture rather
than pretending the original operator already provides it.

Labels are opened only after scores are written. A source-grouped supervised
probe may be run later as a readability ceiling, but it is not an unsupervised
result.

## Evidence rule

The primary control uses the same `HO_GCN`, inputs, high-order nodes, terminal
projection and parameter count, but removes every high-order transition. Thus
`causal - no_transition` isolates the learned value of chronological path
composition. `MODEL=gcn` remains an auxiliary first-order baseline; its
parameter count is reported rather than claimed equal because the copied
architectures have different branches and depths. The order-2 construction is
useful only if causal DBGNN improves frozen node-only detection consistently
over the no-transition control across seeds and tasks.
A supervised probe gain with no unsupervised gain means that label
information is readable but the chosen one-class geometry is not aligned with
hallucination. No gain over GCN means generic causal-walk lifting is not the
missing mechanism.

A layer-time shuffle can subsequently distinguish correct chronology from the
mere presence of high-order transition density; it is not required to establish
the first `causal` versus `no_transition` gate.
