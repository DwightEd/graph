# Causal Typed-Path De Bruijn Routing

This directory is an engineering-isolated, label-free hallucination detector.
Read [`METHOD.md`](METHOD.md) before interpreting its scores.

The workflow is deliberately split into three processes:

```bash
python -m experiments.causal_typed_path_debruijn.main fit \
  --train-split /path/to/train \
  --reference /path/to/output/reference.npz \
  --device cuda

python -m experiments.causal_typed_path_debruijn.main score \
  --test-split /path/to/test \
  --reference /path/to/output/reference.npz \
  --output /path/to/output/test_scores.npz \
  --device cuda

python -m experiments.causal_typed_path_debruijn.main evaluate \
  --test-split /path/to/test \
  --scores /path/to/output/test_scores.npz \
  --output /path/to/output/evaluation.json
```

`fit` and `score` never request labels.  `evaluate` reopens the test split with
evaluation labels enabled only after it freezes and verifies the score file.
The core APIs also require a `train` manifest for fitting and a `test` manifest
for scoring, so swapping the two paths fails before structural computation.
The evaluation report contains overall and per-task token metrics with
whole-response cluster bootstrap intervals.

For a small interface check, use the runner with distinct train and test
limits:

```bash
TRAIN_SPLIT=/path/to/train TEST_SPLIT=/path/to/test \
TRAIN_LIMIT=64 TEST_LIMIT=5 DEVICE=cpu \
OUT=experiments/causal_typed_path_debruijn/outputs/smoke \
bash experiments/causal_typed_path_debruijn/run.sh
```

The repository does not include the real attention cache.  A smoke run checks
runtime and artifact contracts only; it is not evidence for choosing a method
component.

## Module ownership

```text
graph_builder.py       ResearchSample -> channel-preserving sparse graph
layered_automaton.py   layer-unfolded five-state lineage distribution
typed_path_dp.py       finite-horizon near/far path ablation
debruijn.py            label-free soft higher-order transition grammar
change_lockin.py       causal rupture, persistence, and raw channel score
calibration.py         channel ECDF, symmetric fusion, final ECDF
nulls.py               causal endpoint and offline temporal controls
artifacts.py           strict frozen schemas and provenance checks
experiment.py          fit/score orchestration only
evaluation.py          the only label-aware module
visualization.py       per-sample causal graph and phase plots
main.py                CLI only
run.sh                 one-command three-process workflow
```

No file in this directory parses `.pt` or `.npz` attention caches directly.
Raw attention access always goes through the root `research_dataset.py` seam.

Fit requires at least three complete `source_id` groups because grammar fit,
channel calibration, and final fusion calibration are isolated. A production
run should contain substantially more than this minimum.

Optional label-free explanation and the pre-registered RR bridge are separate
commands:

```bash
python -m experiments.causal_typed_path_debruijn.main visualize \
  --test-split /path/to/test \
  --reference /path/to/output/reference.npz \
  --scores /path/to/output/test_scores.npz \
  --sample-id SAMPLE_ID \
  --output /path/to/output/sample.png

python -m experiments.causal_typed_path_debruijn.main hybrid \
  --path-scores /path/to/output/test_scores.npz \
  --rr-scores /path/to/frozen_rr_signal_scores.npz \
  --output /path/to/output/path_rr_hybrid.npz

python -m experiments.causal_typed_path_debruijn.main evaluate \
  --test-split /path/to/test \
  --scores /path/to/output/path_rr_hybrid.npz \
  --output /path/to/output/hybrid_evaluation.json
```

The bridge accepts only `received_topk.causal.residual_tail`; it rejects the
offline response-length-conditioned spectral score.
