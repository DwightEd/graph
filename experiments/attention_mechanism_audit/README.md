# Functional Message Graph

This directory now has one job: convert a teacher-forced response into a graph
of the messages that the frozen observer actually used.  It no longer contains
the v10 four-branch shortcut score family.  The QA and Summary results showed
that the observed shortcut score was not better than its endpoint-rewired
control, so exact response-endpoint pairing is not retained as a mechanism
claim.

## Graph quantity

For response token `t`, the causal predictor is

\[
q_t=P-1+t,
\]

where `P` is the first response position.  At layer `l`, query head `h`, and
causal source `j`, the transported head-space message is

\[
u_{j\to q_t}^{l,h}=A_{q_tj}^{l,h}V_j^{l,\kappa(h)},
\]

and its exact residual-space write is

\[
m_{j\to q_t}^{l,h}=W_{O,l,h}u_{j\to q_t}^{l,h}.
\]

Attention is only the selection coefficient.  `A*V` is the transported
content.  `W_O(A*V)` is what enters the residual stream.

The graph also asks whether that message is functional for the observed token.
Each target is replayed as an independent one-token decoding problem and
differentiated only against its own teacher-forced log probability. Several
independent prefixes are padded into one batch for throughput; they do not
share activations or losses. If

\[
g_{t,l}=\frac{\partial\log p(y_t)}{\partial o_{t,l}},
\]

then the signed edge attribution is

\[
\boxed{c_{j\to t}^{l,h}=g_{t,l}^{\top}m_{j\to q_t}^{l,h}}.
\]

The implementation captures its gradient directly at the input of `o_proj`,
which is exactly `W_O^T g`, and therefore avoids an extra 4096-by-4096 multiply.
It is computed without materializing the 4096-dimensional edge vector:

\[
c_{j\to t}^{l,h}=A_{q_tj}^{l,h}
\left\langle W_{O,l,h}^{\top}g_{t,l},V_j^{l,\kappa(h)}\right\rangle.
\]

Because the attention output is linear in each coefficient, the same quantity is
also

\[
c_{j\to t}^{l,h}=A_{q_tj}^{l,h}
\frac{\partial\log p(y_t)}{\partial A_{q_tj}^{l,h}}.
\]

Thus it is the signed, value-aware form of attention-times-gradient saliency,
not raw attention saliency. Positive values support the observed token locally;
negative values oppose it. It is a first-order functional attribution, not a
claim of exact causal necessity. Hallucination labels are never used during
capture.

## Graph structure

Each Transformer layer has `pre`, `post_attention`, and `post_mlp` state nodes.
An attention message is a typed edge

```text
pre(layer, source token) -> post_attention(layer, predictor token)
```

with its layer, query head, source role, attention coefficient, signed
functional attribution, exact residual-message norm, and the head-space `A*V`
factor that reconstructs the residual message through the frozen `W_O` block.  Residual, MLP,
cross-layer, and output links are stored separately as structural edge types.
Thus a downstream graph method receives the actual computation incidence, not
an invented GNN message function.

Every causal source and every head contributes to `node_profile`:

```text
[response token, layer, head, source role, channel]
```

The four source roles are `evidence`, `other_prompt`, `response_history`, and
`predictor_self`.  The channels are raw attention mass, exact residual-message norm,
positive functional attribution, and negative functional attribution.  No
layer or head is averaged.  The MLP write has an analogous norm/positive/
negative profile.

`node_embedding` is only the flattened typed graph tensor so existing
node-level evaluators can consume it.  It is not a learned feature stack.

## Explicit edges and all-data use

The dense profile is computed before edge selection and therefore uses every
causal source.  The explicit edge list keeps the messages covering a requested
fraction of absolute functional attribution, subject to a storage budget.  The
exact omitted profile is saved as `edge_tail_profile`, so pruning cannot alter
the node representation or silently discard aggregate mass.

`--edge-budget 0` stores every edge.  This can be extremely large.  The default
keeps at most 64 explicit attention edges per target and layer while still
using all edges in the dense representation.

## Files

```text
capture.py   one-token cached replay and exact gradients
graph.py     AVWO attribution, graph incidence, dense profile, sparse view
data.py      exact prompt/evidence token alignment
collect.py   dataset traversal, resume, and serialization
export.py    label-free concatenation of node embeddings
run.py       foreground command line entry point
```

There is no detector, calibration, bootstrap suite, or list of hand-designed
shortcut scores in this directory.  Detection and statistical evaluation are
kept downstream from graph construction.

## Run

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
conda activate research
bash experiments/attention_mechanism_audit/run_all.sh
```

Smoke test:

```bash
python -m experiments.attention_mechanism_audit.run build --limit 2
```

The new artifacts are written under

```text
experiments/attention_mechanism_audit/outputs/Meta-Llama-3.1-8B-Instruct/
  functional_message_graph_v1/{train,test}/
```

The old v10 output directories are not read, adapted, or deleted.
