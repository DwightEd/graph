# CMRP implementation plan

This plan converts `METHOD.md` into a testable first implementation while
keeping the scientific claims narrower than the eventual evidence-flow model.
The work is organized so that every file has one responsibility and no module
acts as a later patch layer.

## 1. Directory contract

```text
experiments/causal_multiplex_flow/
├── METHOD.md                scientific hypothesis and claim boundaries
├── IMPLEMENTATION_PLAN.md   this engineering plan
├── README.md                commands, outputs, and interpretation
├── __init__.py              public API only
├── events.py                canonical sparse attention -> causal events
├── controls.py              deterministic source counterfactuals
├── model.py                 channel-aware dynamic source predictor
├── calibration.py           source-group split and empirical tails
├── artifacts.py             strict schemas and loaders
├── experiment.py            fit, score, evaluation orchestration
├── main.py                  CLI
└── run.sh                   one foreground runner

tests/
└── test_causal_multiplex_flow.py
```

No experiment file may import `cache.py` or `formal_cache.py`.  Test fixtures may
use the canonical cache writers, as existing tests do.

---

## 2. Phase A: exact event extraction

### `events.py`

Input:

```python
sample = dataset[sample_id]
```

Output:

```text
CausalEventSample
  target_ptr             [T+1]
  relation               [E]  RP or RR
  source                 [E]  response-relative; -1 for prompt role
  channel                [E]  layer * num_heads + head
  weight                 [E]
  lag                    [E]  positive for RR, zero for RP
  full_role_summary      [T,4]
```

The four role summaries are full retained:

```text
RP mass, RR mass, RP edge count, RR edge count
```

Typed top-k selection is applied only to the event list consumed by the neural
encoder.  Full summaries are computed before selection.

Required invariants:

- `source < target` for every RR event;
- `target_ptr` partitions all selected events exactly once;
- channel indices remain in `[0, L*H)`;
- event ordering is deterministic;
- no prompt relative-position feature is created;
- missing CSR edges are never materialized.

---

## 3. Phase B: topology counterfactuals

### `controls.py`

Implement deterministic candidate construction for every selected RR event.
The candidate set contains:

1. true source at index 0;
2. one lag-bin-preserving alternative when possible;
3. additional unique negatives, prioritizing the same lag bin;
4. fallback prior sources only when the lag bin lacks enough alternatives.

The source-rewiring control must preserve:

```text
target, channel, relation, retained weight, and coarse log2 lag bin
```

It must change the exact source whenever a legal alternative exists.

Functions remain pure: no labels, no global random state, no mutation of the
canonical sample.

---

## 4. Phase C: dynamic multiplex model

### `model.py`

Implement:

```text
CausalMultiplexRouter
```

Core components:

- learned layer embeddings;
- learned head embeddings;
- learned RP/RR relation embeddings;
- a generic prompt-anchor state;
- Fourier causal-lag features;
- shared event-message MLP;
- mean/max event pooling;
- role-summary MLP;
- GRU token-state update;
- RR-presence head;
- source-query/source-candidate projection heads;
- retained edge-weight diagnostic head.

The model processes one response sequentially.  Before token `t` consumes its
incoming events, `h_{t-1}` predicts retained-RR presence and scores candidate
sources for selected RR edges.  The events are then encoded to produce `h_t`.

Returned per-token arrays:

```text
raw_route_surprise
presence_nll
source_nll
weight_error
rewired_source_nll
rewire_gap
selected_rr_edges
```

Returned edge-level topology diagnostic:

```text
rewire_edge_gap
```

Each finite entry is the raw source NLL difference for one evaluated rewired
RR edge. Source cross-entropy terms are raw NLLs, without candidate-count
normalization.

Returned scalar training loss:

```text
mean(raw_route_surprise) + weight_loss_weight * mean(weight_error)
```

The primary score remains `raw_route_surprise`; weight error is diagnostic.

---

## 5. Phase D: split, calibration, and artifacts

### `calibration.py`

- derive one group key from `source_id` or `sample_id`;
- split complete groups deterministically into fit and calibration streams;
- ensure the streams are non-empty and disjoint;
- implement a finite-sample monotone empirical upper-tail score;
- never use token labels or test data.

### `artifacts.py`

Schemas:

```text
cmrp-reference-v2
cmrp-score-v2
cmrp-evaluation-v2
```

Reference files store:

- geometry and all configs;
- model file name and SHA-256;
- fit/calibration group IDs;
- calibration raw-route-surprise distribution;
- edge-level calibration topology-gate fields
  `topology_gate_evaluated_edge_count`,
  `topology_gate_selected_edge_count`, `topology_gate_coverage`,
  `topology_gate_mean_gap`, `topology_gate_median_gap`,
  `topology_gate_positive_fraction`, and `topology_gate_pass`.

Score files store:

- row identifiers and metadata;
- calibrated primary `score`;
- all raw diagnostic score arrays;
- reference/model digests;
- the on-disk test dataset-manifest digest and per-row `response_length`;
- the frozen source audit: `fit_group_id`, `calibration_group_id`,
  `test_group_id`, `test_sample_id`, and `audit_scope`;
- no hallucination labels.

Strict loaders fail on schema, dimensional, digest, or finite-value mismatch.

---

## 6. Phase E: experiment orchestration

### `experiment.py`

#### `fit_cmrp(...)`

1. open only the unlabeled train dataset supplied by the caller;
2. split complete source groups into fit/calibration;
3. train the model on fit samples;
4. freeze parameters;
5. score calibration samples;
6. fit the monotone score calibration;
7. evaluate the true-versus-rewired topology gate without labels;
8. save `model.pt` and `reference.npz`.

#### `score_cmrp(...)`

1. load and validate the model/reference pair;
2. score every selected test sample without labels;
3. convert raw route surprise through the calibration upper tail;
4. freeze `test_scores.npz`.

#### `evaluate_cmrp(...)`

1. capture the score artifact path and SHA-256;
2. load the artifact only through that captured path;
3. reverify the artifact digest, exact dataset-manifest digest, and expected
   test split;
4. verify canonical source, attention-derived response length, and complete
   `0..R-1` token coverage for every scored response;
5. open the evaluation label store only now and align by
   `(sample_id, token_index)`;
6. report AUROC/AUPRC for the primary and diagnostics;
7. write `evaluation.json`.

The score artifact follows the generic `score*` convention so it can be used by
`experiments/conditioned_benchmark/` without a method-specific evaluator.

---

## 7. CLI and runner

### `main.py`

Commands:

```text
fit
score
evaluate
```

### `run.sh`

Environment variables:

```text
ROOT
OUT
DEVICE
TRAIN_LIMIT
TEST_LIMIT
EPOCHS
MAX_RP_EVENTS
MAX_RR_EVENTS
NEGATIVES
HIDDEN_DIM
SEED
```

A meaningful smoke test must use enough training source groups to create
separate fit and calibration sets.  The runner should reject a tiny fit before
silently training an uninterpretable model.

Default smoke command:

```bash
TRAIN_LIMIT=64 TEST_LIMIT=5 EPOCHS=1 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_multiplex_flow/run.sh
```

Full command:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_multiplex_flow/run.sh
```

Smoke and full outputs must be isolated.

---

## 8. Required tests

### Event tests

- exact RR source/channel/lag preservation;
- prompt represented only as RP role, not relative position;
- typed top-k deterministic selection;
- role summaries use all retained edges before selection;
- no non-causal RR source survives.

### Counterfactual tests

- candidate index 0 is always the true source;
- rewired source is different when an alternative exists;
- rewired source remains legal and in the same log-lag bin;
- repeated calls with the same seed are identical.

### Model tests

- output shapes equal response length;
- every returned loss is finite;
- backward pass reaches event/channel and dynamic parameters;
- first token without RR history is handled by the presence objective;
- no labels are accepted as model input.

### End-to-end contract test

Synthetic canonical datasets run:

```text
fit -> score -> evaluate
```

and verify:

```text
fit labels_read = false
score labels_read = false
reference contains no label/y_token
score artifact contains no label/y_token
primary score equals calibrated raw_route_surprise
labels open only during evaluate
```

The existing experiment data-contract suite must also pass.

---

## 9. Validation before merge

The branch must pass:

```text
python -m py_compile for every new module
bash -n experiments/causal_multiplex_flow/run.sh
python -m unittest tests.test_causal_multiplex_flow -v
python -m unittest tests.test_experiment_data_contract -v
python -m unittest tests.test_research_views -v
```

A temporary pull-request workflow may be used for validation and must be removed
before merge.  The final merge should be based on the then-current `main`, not
on a stale branch if the repository changes during implementation.

---

## 10. Deferred work, not hidden TODOs

The first merge intentionally does not include:

- semantic prompt-role extraction;
- full censored no-edge likelihood;
- spectral-consistency training;
- temporal-WL/event-graph signatures;
- channel-consistent per-head hidden states;
- supervised head selection;
- test-label score fusion.

These extensions require separate hypothesis gates and should not be inserted
as untested branches inside the Phase-1 detector.
