# CITG implementation plan

This plan translates `METHOD.md` into a falsifiable first implementation. Every
file has one responsibility; the former CMRP neural predictor is removed rather
than retained as an inactive patch path.

## 1. Directory contract

```text
attention_graph/
├── causal_events.py          reusable canonical event extraction
└── topology_controls.py      reusable deterministic rewires

experiments/causal_isomorphism_trajectory/
├── METHOD.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── __init__.py
├── signatures.py             temporal invariant + two-axis features
├── geometry.py               conditioned PPCA and topology gate
├── artifacts.py              strict frozen schemas
├── experiment.py             fit/score/evaluate orchestration
├── main.py                   CLI
└── run.sh                    foreground runner

tests/
└── test_causal_isomorphism_trajectory.py
```

The old `experiments/causal_multiplex_flow/` neural detector and its test are
removed. Its negative result is preserved in
`docs/results/cmrp_negative_result.md`.

## 2. Shared causal events

`attention_graph/causal_events.py` must:

- consume only `ResearchSample.iter_sparse_attention_blocks()`;
- preserve exact RR source/target, layer, head, weight and lag;
- represent prompt sources only as relation `RP` and source `-1`;
- compute full role and layer-band summaries before selection;
- select events deterministically per target, relation and layer band;
- never materialize missing CSR entries;
- validate strict causality and tensor geometry.

## 3. Rewiring controls

`attention_graph/topology_controls.py` must:

- rewire only RR events;
- keep target/layer/head/relation/weight/coarse-lag-bin fixed;
- change source whenever a legal same-bin alternative exists;
- return edge- and token-level availability masks;
- use deterministic SHA-based selection;
- never mutate canonical dataset objects.

## 4. Temporal invariant signatures

`signatures.py` implements:

- one-hop rooted event-label histograms;
- time-respecting two-hop path-label histograms;
- source-sharing motifs;
- count and weight hashing;
- global and ordered layer-band signatures;
- RP/RR role and concentration features;
- generation-time deltas;
- adjacent-band depth distances and late-depth summaries;
- four preregistered variants: full/static/topology/mass.

No final response length or prompt relative-position moment is allowed.

## 5. Unlabeled geometry

`geometry.py` implements:

- task × causal-position condition keys;
- robust per-condition median/MAD with global fallback;
- fit-only PCA/PPCA geometry;
- disjoint calibration upper-tail mapping;
- source-group bootstrap for the rewire-energy topology gate.

The primary score is PPCA energy of the full trajectory. No component weights
or direction are chosen from labels.

## 6. Artifacts and provenance

Schemas:

```text
citg-reference-v1
citg-score-v1
citg-evaluation-v1
```

Reference artifacts store:

- dataset-manifest digest;
- event/signature/geometry configs;
- fit/calibration source groups;
- all four fitted geometry models;
- all four calibration energy distributions;
- source-bootstrap topology gate.

Score artifacts store:

- complete token rows and metadata;
- primary and preregistered ablation scores;
- raw energies and rewire diagnostics;
- reference digest/path;
- test manifest and source audit;
- no labels.

## 7. Execution stages

### `fit_citg`

1. partition complete train source groups;
2. sample fit-token trajectories;
3. fit four condition-aware geometries;
4. score every calibration token;
5. compute true/rewired gate;
6. freeze `reference.npz`.

### `score_citg`

1. verify reference;
2. audit held-out test source groups;
3. extract true and rewired trajectories;
4. freeze primary/ablation scores;
5. write `test_scores.npz`.

### `evaluate_citg`

1. capture score file identity and digest;
2. strict-load and verify reference provenance;
3. verify exact test manifest and complete token rows;
4. open labels;
5. report fixed-direction AUROC/AUPRC.

## 8. Required tests

- event extraction preserves exact source/layer/head/lag;
- selection is band-balanced and deterministic;
- no centroid field exists;
- rewire preserves coarse lag and channel;
- shifted but isomorphic rooted motifs have equal signatures;
- rewired topology changes the signature when eligible;
- condition scaler and PPCA are finite;
- full synthetic `fit -> score -> evaluate`;
- fit/score artifacts contain no labels;
- source-group audits are disjoint;
- existing research-view and data-contract tests pass.

## 9. Validation and merge

Before merge:

```text
python -m py_compile all new modules
bash -n run.sh
tests.test_causal_isomorphism_trajectory
tests.test_experiment_data_contract
tests.test_research_views
```

A temporary PR workflow is allowed and must be deleted after success.

## 10. Deferred work

Not hidden inside Phase 1:

- semantic prompt-role segmentation;
- full censored-edge likelihood;
- exact temporal WL/event-graph canonicalization;
- nonlinear density estimation;
- supervised layer/head selection;
- test-label direction inversion or score fusion.
