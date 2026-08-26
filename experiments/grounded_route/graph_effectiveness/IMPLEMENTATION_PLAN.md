# Implementation Plan

## Public flow

```text
verify(test/index.npz)
  -> integrity.json
  -> label_free_topology.npz

audit(calibration/index.npz, test/index.npz, canonical test split)
  -> freeze and verify saved embedding bundles
  -> fit representation detectors and a position nuisance baseline without labels
  -> freeze all test anomaly/control scores
  -> open canonical labels for post-hoc metrics
  -> run source-grouped node-only readability probes
  -> oof_predictions.npz + report.json
```

Optional control bundles are complete outputs from independently run
`no_message`, `endpoint_rewire` or `weight_shuffle` pipelines. They are
row-aligned with the real bundle and pass through identical node-only readers.

## Modules

```text
data.py         index/sidecar SHA, invariant and row verification; label gate
views.py        saved-edge mechanism views and encoded-bundle row alignment
model.py        node-only MLP, autoencoder and Deep-SVDD networks
detectors.py    fixed label-free detector suite over calibration embeddings
label_free.py   neighbour-alignment mechanism sanity check
metrics.py      AUROC/AUPRC and paired source-cluster bootstrap
upper_bound.py  source-disjoint linear/MLP node-representation probes
audit.py        score freezing, post-hoc evaluation and variant comparison
run.py          verify/audit CLI
run.sh          cluster entry point over an existing GroundedRoute output
```

## Artifact contracts

```text
integrity.json
  label-free sidecar, alignment and conservation facts

label_free_topology.npz
  complete [layer,head] neighbour-alignment tensors; not detector features

unsupervised_scores.npz
  row identity + frozen representation and position-control scores; no labels

oof_predictions.npz
  row identity + source-grouped node-only probe predictions; no labels

report.json
  prevalence, metrics, paired variant deltas and evidence-scope statements
```

## Engineering constraints

- Representation models receive only `[node, embedding_dimension]` arrays.
- No representation detector or supervised representation probe imports
  `edge_index` or performs message passing; named position baselines are
  isolated nuisance controls.
- Prompt embeddings remain in graph sidecars for integrity checks; only
  response embeddings are scored and labelled.
- Calibration data are source-disjoint from encoder fitting and the test set.
- Test labels never select an unsupervised detector, seed or hyperparameter.
  Supervised probes select epochs only on an inner source-disjoint split.
- Outer and inner supervised folds are source-disjoint.
- Variant bundles must have identical sample/token rows.
- PyTorch and scikit-learn already required by GroundedRoute are the only model
  dependencies.

## Verification

Tests cover sidecar identity/alignment, mass conservation, deterministic
mechanism controls, variant row alignment, detector output shapes, source-fold
disjointness, label isolation and a synthetic node-embedding anomaly problem.
Repository checks also include Python compilation, shell syntax and
`git diff --check`.
