# Design audit

## Rejected design

The former implementation stored a `PrefixState` and emitted 39 handcrafted
statistics. It briefly formed `[source, layer, head]` tensors, but most analyses
then averaged them into mass, counts, cosine similarities, or heuristic ranks.
That code could not learn message propagation and could not test whether the full
layer-head graph was recoverable.

## Current hypothesis

A useful attention-graph method requires all three conditions:

1. full layer-head values recover better than layer/head averages;
2. exact source endpoints and neighbor messages improve recovery;
3. recovery or structural reliance differs between correct and hallucinated
   tokens after position and graph-density matching.

No direction is assumed. Hallucinated graphs may be harder to recover, easier to
recover as a stable erroneous regime, or indistinguishable.

## Canonical representation

The indivisible observation is

```text
(source, target, layer, head, weight)
```

Sparse observations are materialized as one exact token-pair edge with an
`[L, H]` tensor and observation mask. This is at least as information-preserving
as CHARM's flattened `L*H` edge vector, while keeping layer and head axes explicit.

## Learned operator

At each transformer layer, the model reads the complete head vector, sends
messages along exact token-pair endpoints, updates token states, and reconstructs
masked channels. This is explicit layer-ordered graph propagation. It does not
use handcrafted prompt paths, source popularity, source co-use counts, or
manually weighted recovery formulas.

## Required gates

The graph claim is rejected when any of these fail:

- `full_channel_gain <= 0`: full channels do not beat a global mean input;
- `layer_head_gain <= 0`: head structure does not beat per-layer means;
- `layer_order_gain <= 0`: ordered layers do not beat shuffled layers;
- `message_gain <= 0`: neighbor propagation does not help;
- `endpoint_gain <= 0`: exact endpoints do not beat matched rewiring.

Only after these gates pass is it meaningful to use learned graph embeddings in
an unsupervised hallucination method.
