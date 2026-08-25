# HoloRoute

HoloRoute learns normal structure in an attention event graph without
hallucination labels.

Read the code in this order:

```text
graph.py       attention -> event graph
model.py       event graph -> neural event states
learning.py    self-supervised graph tasks
detection.py   local residuals -> anomaly score
pipeline.py    train, calibrate and score
evaluate.py    label-posthoc metrics
baseline.py    flat all-layer no-topology control
run.py         command-line interface
```

The public computation path is deliberately short:

```python
graph = build_graph(sample, config.graph)
model = HoloRoute(graph.layer_count, graph.head_count, config.model)
loss = self_supervised_loss(model, graph, config, generator)
residuals = score_graph(model, graph, config, seed)
```

Internal tensors are grouped in small data classes. The model returns one
`ModelOutput` rather than a dictionary of named feature fields. Token residuals
are stored as one `[token, residual]` matrix; names exist only in the reporting
layer.

## Run

```bash
TRAIN_SPLIT=/path/to/train \
TEST_SPLIT=/path/to/test \
OUT=experiments/holoroute/outputs/full \
MODEL=holoroute DEVICE=cuda \
bash experiments/holoroute/run.sh
```

Flat all-layer control:

```bash
TRAIN_SPLIT=/path/to/train \
TEST_SPLIT=/path/to/test \
OUT=experiments/holoroute/outputs/flat \
MODEL=flat1024 DEVICE=cuda \
bash experiments/holoroute/run.sh
```
