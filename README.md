# HoloRoute

HoloRoute is the active research implementation in this repository. It performs attention-only, unsupervised, token-level hallucination detection by learning normal structure on a multilayer causal attention event graph.

The default branch has been consolidated around one method. Historical implementations are no longer kept as executable code on `main`; their hypotheses, results, and rejection decisions are recorded in [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md). Historical branch refs are archival only and are not active development lines.

## Method in one line

```text
sparse internal attention
-> one event graph per prompt-response sample
-> self-supervised graph completion
-> position-conditioned one-class score
-> token-level evaluation
```

Each graph node is an exact `(source token, target token, layer)` attention event whose attribute is the complete head profile at that layer. The graph contains:

- depth edges for the same token pair across adjacent Transformer layers;
- causal relay edges for ordered paths `u -> s -> t`;
- query groups containing all events entering one target-layer pair;
- depth/relay diamonds used to audit transport composition.

See [`experiments/holoroute/METHOD.md`](experiments/holoroute/METHOD.md) and [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md).

## Repository layout

```text
cache.py                       canonical sparse attention cache
formal_cache.py                adapter for formal PT attention caches
research_dataset.py            shared data interface
experiment_protocol.py         source-group and evaluation protocol
experiments/holoroute/         active method, baseline, runners, and tests
docs/EXPERIMENT_HISTORY.md     prior experiments and recorded results
docs/RESEARCH_STATUS.md        current claims, gates, and next experiments
```

## One-command QA run

The server-specific runner uses the current RAGTruth attention cache paths:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
bash experiments/holoroute/run_qa.sh
```

The generic entry point is:

```bash
TRAIN_SPLIT=/path/to/attention/train \
TEST_SPLIT=/path/to/attention/test \
OUT=experiments/holoroute/outputs/qa \
TASK=QA MODEL=holoroute DEVICE=cuda EPOCHS=8 \
bash experiments/holoroute/run.sh
```

The Flat-1024 no-topology control uses the same all-layer, all-head values:

```bash
TRAIN_SPLIT=/path/to/attention/train \
TEST_SPLIT=/path/to/attention/test \
OUT=experiments/holoroute/outputs/flat_qa \
TASK=QA MODEL=flat1024 DEVICE=cuda EPOCHS=8 \
bash experiments/holoroute/run.sh
```

## Outputs

```text
model.pt
reference.npz
scores.npz
evaluation/evaluation.json
evaluation/position.csv
evaluation/residuals.csv
```

Training, calibration, and scoring do not read hallucination labels. Labels are opened only by the evaluation command after the score artifact has been frozen.

## Tests

```bash
python -m compileall -q experiments/holoroute
bash -n experiments/holoroute/run.sh
pytest -q experiments/holoroute/tests
```

The current method is an implemented research prototype, not yet a validated SOTA result. The mandatory acceptance tests are documented in [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md).
