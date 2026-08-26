# GroundedRoute

GroundedRoute is the active representation-learning experiment in this
repository. It learns one label-free embedding for every token from the exact
typed attention graph, then freezes those embeddings before applying an
independent unsupervised detector.

HoloRoute remains in [`experiments/holoroute/`](experiments/holoroute/) as the
masked event-reconstruction baseline. Its reconstruction residual is not the
GroundedRoute representation or score.

## Method in one line

```text
sparse internal attention
-> one typed token graph per prompt-response sample
-> conserved prompt / response-closed lineage
-> causal exact-endpoint prediction
-> frozen token embeddings
-> embedding-only one-class detector
-> token-level evaluation
```

The node is a token. Every retained attention trace remains an explicit
`(source, target, layer, head, weight)` edge; layer/head identity is not averaged
before message passing. A learned row-stochastic head transition propagates a
conserved three-state lineage: prompt-origin, response-closed, and unresolved.

The same pipeline can retrain `real`, `weight_shuffle`, and
`endpoint_rewire` variants under one frozen split, seed, budget, and detector.
Control artifacts record their actual changed-edge fraction; endpoint rewiring
matches only a coarse logarithmic lag bucket, not exact lag.

See [`experiments/grounded_route/README.md`](experiments/grounded_route/README.md)
and [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md).

## Repository layout

```text
cache.py                       canonical sparse attention cache
formal_cache.py                adapter for formal PT attention caches
research_dataset.py            shared data interface
experiment_protocol.py         source-group and evaluation protocol
experiments/grounded_route/    active token-graph representation experiment
  graph_effectiveness/         node-only detector/construction audit
experiments/dbgnn_reference/   original-code order-2 DBGNN vs GCN reference
experiments/holoroute/         masked reconstruction baseline
docs/EXPERIMENT_HISTORY.md     prior experiments and recorded results
docs/RESEARCH_STATUS.md        current claims, gates, and next experiments
```

## One-command QA run

The complete label-free training, encoding, detection, and post-hoc evaluation
pipeline is:

```bash
TRAIN_SPLIT=/path/to/attention/train \
TEST_SPLIT=/path/to/attention/test \
VARIANT=real OUT=experiments/grounded_route/outputs/qa/real \
TASK=QA DEVICE=cuda EPOCHS=8 \
bash experiments/grounded_route/run.sh
```

## Outputs

```text
train_graph.json          label-free dataset selection
model.pt                  causal token-graph encoder
calibration/graphs/*.pt   calibration graph + node embeddings
calibration/index.npz     detector-reference embeddings
test/graphs/*.pt          per-sample graph + node embeddings
test/index.npz            test response-node embeddings
detector.npz              PCA-whitened kNN reference
scores.npz                one scalar per response token
evaluation.json           post-hoc metrics
```

The saved `node_embedding` is already the output of typed message passing. The
[`graph_effectiveness`](experiments/grounded_route/graph_effectiveness/)
subpackage tests it with node-only unsupervised detectors and separately
encoded graph controls; it does not attach another GNN after the representation.

Training, calibration, and scoring do not read hallucination labels. Labels are opened only by the evaluation command after the score artifact has been frozen.

To test whether generic causal-walk lifting helps independently of
GroundedRoute's mechanism, [`dbgnn_reference`](experiments/dbgnn_reference/)
uses the paper authors' core `HO_GCN/GCN` code on the same saved token graphs.
It exports the pre-classifier token state and compares order-2 DBGNN with a
matched first-order GCN using embedding-only detectors.

## Tests

```bash
python -m compileall -q experiments/grounded_route
bash -n experiments/grounded_route/run.sh
pytest -q experiments/grounded_route/tests
pytest -q experiments/grounded_route/graph_effectiveness/tests
pytest -q experiments/dbgnn_reference/tests
bash -n experiments/dbgnn_reference/run.sh \
  experiments/dbgnn_reference/run_compare.sh
```

The implementation is a research prototype, not a validated SOTA result. Its
graph claim is conditional on the matched topology controls and held-out
token-level results in [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md).
