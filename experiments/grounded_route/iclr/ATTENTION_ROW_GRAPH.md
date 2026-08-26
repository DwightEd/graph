# Attention-Row Graph: layer/head-aware token representation

## Why this increment exists

The averaged first-order GCN is the first graph baseline in this project that
produced a clearly useful node representation:

```text
PCA-kNN        AUROC 0.6982  AUPRC 0.1617
linear probe   AUROC 0.7865  AUPRC 0.2999
```

Its weakness is equally clear: all Transformer layers and heads are averaged
before message passing. The next experiment therefore keeps the successful
one-hop token graph idea, but changes only the two parts that matter most:

1. how an attention neighbourhood is aggregated into a token;
2. what the label-free objective asks the representation to preserve.

The method does not use hallucination labels.

## Graph

A prompt-response sample is one causal token graph. A retained attention entry
is an incidence

```text
(source token, target token, layer, head, weight)
```

For every `(target, layer, head)`, the complete weighted source set is treated
as one attention-row hyperedge. The implementation does not materialize
separate hyperedge nodes; it performs the equivalent source-set aggregation
directly.

## Source-set aggregation

For source messages `x_s` and attention weights `a_s`, prompt sources and
response sources are kept separate. For each group the encoder computes:

\[
m=\sum_s a_s,
\qquad
\mu=\frac{\sum_s a_s x_s}{m+\epsilon},
\qquad
\sigma=\sqrt{\frac{\sum_s a_s x_s^2}{m+\epsilon}-\mu^2}.
\]

The row representation contains:

```text
prompt weighted mean
prompt weighted spread
prompt retained mass
response weighted mean
response weighted spread
response retained mass
self diagonal message
unresolved sparse mass
```

This avoids three problems of an unnormalised sum:

- node degree does not directly set the message scale;
- prompt and response routes cannot cancel each other in one pool;
- conflicting neighbours are distinguishable from a coherent neighbourhood.

Heads are pooled only after their row representations have been formed.
Transformer layers are processed in order, and a GRU updates each response
token after every layer. The final detector receives only the resulting
`node_embedding`.

## Label-free training objective

The old pairwise objective asks only whether one real endpoint outranks one
matched non-edge. The new objective predicts an entire sampled attention row.

For a selected row `(t,l,h)`, the candidate set contains:

- every retained real source in that row;
- causal non-edges matched by source role and logarithmic lag.

The model produces one score per candidate. A softmax is taken within the row.
The target probability of a real source is its normalised retained attention
weight:

\[
q(s\mid t,l,h)
=
\frac{A_{t,s}^{l,h}}
{\sum_{j\in\mathcal S_{t,l,h}}A_{t,j}^{l,h}}.
\]

The route loss is:

\[
\mathcal L_{\text{row}}
=
-\frac1{|\mathcal R|}
\sum_{r\in\mathcal R}
\sum_{s\in\mathcal S_r}
q_r(s)\log p_r(s).
\]

Matched negatives have target probability zero. A small variance regulariser
prevents a constant embedding.

This objective preserves three things that independent pairwise ranking does
not explicitly preserve:

```text
the complete source coalition
relative attention weight inside the row
the identity of layer and head
```

## What is and is not new

The neural components are standard: MLPs, weighted moments, head pooling and a
GRU. The research proposal is the representation unit and the objective:

```text
attention row = weighted source-set hyperedge
row distribution prediction = label-free representation objective
```

The experiment must show that this choice is better than:

```text
averaged first-order GCN
pairwise endpoint objective
no neighbour messages
matched endpoint rewiring
matched weight shuffling
position-only readers
```

Without those gains, the method remains a clean baseline rather than a paper
contribution.

## Run QA

Train and export the new representation:

```bash
bash experiments/grounded_route/run_attention_row_qa.sh
```

Run the same node-only detector/probe suite used for the GCN result:

```bash
bash experiments/grounded_route/evaluation/run_attention_row_qa.sh
```

The report is written to:

```text
experiments/grounded_route/outputs/qa_attention_row/evaluation/report.json
```

## Decision rule

Keep this direction only if at least one of the following is observed on the
same source-disjoint test tokens:

1. unsupervised PCA-kNN or Isolation Forest exceeds the averaged GCN result;
2. the supervised linear readability ceiling exceeds the averaged GCN result;
3. `real` is significantly better than `no_message`, endpoint rewire and
   weight shuffle under paired source bootstrap.

A lower training loss by itself is not evidence that the graph construction is
better.
