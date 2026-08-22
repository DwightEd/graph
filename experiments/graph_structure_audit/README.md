# Multiplex Graph Recovery Audit

This experiment asks a narrow question before designing a hallucination detector:

> Does the full causal attention graph have a recoverable layer-head structure,
> and do correct and hallucinated response tokens differ in that recoverability?

The previous audit reduced the graph to dozens of handcrafted prefix statistics.
Those statistics are removed from the main path. The current implementation keeps
all retained layer-head values and learns recovery directly from the graph.

## Sample graph

One `prompt + response` sample is one graph. A unique token pair is an edge

```text
source token s -> response token t
```

with an attribute tensor

```text
edge_attr[e, layer, head]  # [E, L, H]
```

and a matching observation mask. Response-node diagonal attention is stored as

```text
diagonal[node, layer, head]  # [N, L, H]
```

No layer, head, source, or target averaging is performed when the graph is
materialized. Sparse layer-head events are grouped only into their exact token-pair
edge tensor.

## Learned recovery

Random active channels, complete pair-layer slices, and response-node diagonal
channels are masked. The model then walks through transformer depth:

1. encode the full head vector for every edge at layer `l`;
2. send source-node messages along exact token-pair endpoints;
3. aggregate messages at response targets;
4. update token states with a GRU cell;
5. reconstruct the masked head vector and node diagonal for layer `l`.

The model therefore represents the graph as an ordered sequence of layer-specific
message-passing graphs, rather than a bag of 1024 values or a list of scalar
statistics.

## Structural controls

The same frozen mask is scored under:

- `no_message`: removes neighbor propagation;
- `layer_shuffled`: destroys transformer-depth order;
- `head_shuffled`: destroys head identity;
- `endpoint_rewired`: keeps edge tensors but changes exact source endpoints;
- `layer_mean`: keeps layer means but removes head structure;
- `global_mean`: removes both layer and head structure.

Positive gains mean the corresponding structure improves recovery. These are
model-reliance tests, not hallucination labels.

## Frozen token scores

Each response token receives:

```text
recovery
edge_recovery
diagonal_recovery
message_gain
layer_order_gain
head_identity_gain
endpoint_gain
layer_head_gain
full_channel_gain
```

Training and scoring do not read hallucination labels. Evaluation opens labels
only after scores are saved and reports source-level bootstrap intervals and
same-response matched effects.

## Run

```bash
ROOT=/path/to/attention_cache \
OUT=experiments/graph_structure_audit/outputs/smoke \
TRAIN_LIMIT=30 TEST_LIMIT=10 EPOCHS=2 SCORE_ROUNDS=2 DEVICE=cpu \
  bash experiments/graph_structure_audit/run.sh
```

For the full audit:

```bash
ROOT=/path/to/attention_cache DEVICE=cuda \
  bash experiments/graph_structure_audit/run.sh
```

The main result is not assumed in advance. `recoverability.csv` reports whether
correct tokens are more recoverable, hallucinated tokens are more recoverable,
or the result is inconclusive. `structure_gates.csv` reports whether message
passing, exact endpoints, layer order, head identity, and full layer-head values
actually improve recovery.
