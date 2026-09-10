# Constraint-Control Graph Anomaly Detection

This repository tests one narrow mechanism hypothesis: a hallucination can start
after the model has recovered relevant content but failed to preserve the
source constraint's control over the emitted fact.

The active method represents each factual commitment event as a fixed-role,
signed causal graph with exactly four measured edges. It does not train an
attention autoencoder, predict the next layer, or use hallucination labels to
construct or score graphs.

## Execution path

```text
factorial margins JSONL
  -> ControlGraphBuilder.build()
  -> canonical graph JSONL
  -> GraphAnomalyDetector.fit()/score()
  -> frozen anomaly scores
  -> DetectionEvaluator.run() + separate labels
```

The single entry point is `main.py`:

```bash
python main.py build \
  --input data/factorial_events.jsonl \
  --output outputs/graphs.jsonl

python main.py detect \
  --graphs outputs/graphs.jsonl \
  --output outputs/detection \
  --fit-split train \
  --score-split test

python main.py evaluate \
  --scores outputs/detection/scores.jsonl \
  --labels data/test_labels.jsonl \
  --output outputs/evaluation.json
```

The shell wrapper runs the label-free build and detection stages:

```bash
bash scripts/run_graph_anomaly.sh
```

With explicit research input and output paths:

```bash
bash scripts/run_graph_anomaly.sh \
  data/factorial_events.jsonl outputs/run-001 train test
```

The no-argument form uses `data/pilot_factorial_events.jsonl`, a synthetic
pipeline smoke test rather than an experiment result.

It refuses to overwrite existing artifacts. Evaluation is deliberately a
separate command so labels cannot enter graph construction or calibration.

## Input contract

Each input line is one independently identified factual event:

```json
{
  "schema": "control-graph/factorial-event@1",
  "event_id": "question-17/fact-0",
  "source_id": "question-17",
  "split": "train",
  "relation": "temporal",
  "margins": {
    "onset_a": 1.4,
    "onset_b": -1.1,
    "world_a_after_a": 1.2,
    "world_a_after_b": 0.3,
    "world_b_after_a": -0.4,
    "world_b_after_b": -1.3
  }
}
```

Every number is an `A minus B` answer-token logit margin. `onset_a` and
`onset_b` are measured before a generated prefix under source worlds A and B.
The other four values form the source-world by generated-prefix 2x2 factorial
intervention. RAGTruth labels alone cannot produce this record; the paired
worlds, answer candidates, commitment position, and prefixes must first be
defined and measured.

## Files

- `main.py`: argument parsing and dispatch only.
- `control_graph/data.py`: strict label-free factorial-event input.
- `control_graph/graph.py`: four causal estimands and graph serialization.
- `control_graph/detector.py`: relation-conditional robust anomaly scoring.
- `control_graph/pipeline.py`: build and detect file workflows.
- `control_graph/evaluation.py`: label-only post-hoc AUROC/AUPRC evaluation.
- `docs/METHOD.md`: estimands, claims, confounds, and experiment gates.
- `docs/EXPERIMENT_HISTORY.md`: retained positive and negative findings.

## Tests

```bash
python -m pytest -q
```

The current implementation is a method scaffold, not a reported hallucination
detector result. A claim requires measured factorial events, source-disjoint
splits, frozen thresholds, task/relationship controls, and multiple models.
