# GroundedRoute

GroundedRoute turns one prompt-response sample into a typed causal token graph
and learns one label-free representation per token. Downstream detectors read
only the saved `node_embedding`; they do not run another GNN.

Coding rules for this project are fixed in
[`iclr/ENGINEERING_RULES.md`](iclr/ENGINEERING_RULES.md).

## Current graph method

Each retained attention entry keeps:

```text
source token
target token
Transformer layer
attention head
attention weight
```

The current increment treats every `(target, layer, head)` attention row as a
weighted source-set hyperedge. Prompt and response neighbours are aggregated
separately with weighted mean, weighted spread and retained mass. Diagonal and
unresolved sparse mass are kept as separate inputs. Heads are pooled after the
row representation is formed, and token states are updated layer by layer.

The historical objective ranks one real endpoint against matched non-edges.
The new attention-row experiment predicts the complete retained source
distribution of a sampled row, with target probabilities proportional to the
attention weights.

Detailed method description:

```text
iclr/ATTENTION_ROW_GRAPH.md
```

## Core files

```text
graph.py          sparse attention -> typed token graph
aggregation.py    prompt/response source-set moments
lineage.py        prompt / response-closed / unresolved path state
model.py          layer-wise token encoder
learning.py       pairwise and row-distribution label-free objectives
pipeline.py       build / fit / encode / detect
evaluation/       node-only detectors, probes and construction controls
```

## Existing averaged-GCN reference

`experiments/dbgnn_reference/` contains the simple averaged first-order GCN and
the order-2 DBGNN comparison. The GCN is a baseline, not the active
layer/head-aware method.

## Run the attention-row experiment

```bash
bash experiments/grounded_route/run_attention_row_qa.sh
```

Run the same node-only evaluation suite used for the GCN result:

```bash
bash experiments/grounded_route/evaluation/run_attention_row_qa.sh
```

Outputs:

```text
experiments/grounded_route/outputs/qa_attention_row/
├── model.pt
├── calibration/index.npz
├── calibration/graphs/*.pt
├── test/index.npz
├── test/graphs/*.pt
├── detector.npz
├── scores.npz
└── evaluation/report.json
```

## Construction controls

The full control run independently trains and encodes:

```text
real
no_message
endpoint_rewire
weight_shuffle
```

```bash
bash experiments/grounded_route/evaluation/run_attention_row_controls_qa.sh
```

## Tests

```bash
python -m compileall -q experiments/grounded_route
bash -n experiments/grounded_route/run.sh
pytest -q experiments/grounded_route/tests
pytest -q experiments/grounded_route/evaluation/tests
```
