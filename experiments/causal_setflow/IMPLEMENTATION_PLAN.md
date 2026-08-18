# MG-CASF implementation plan

## 1. Reuse and retirement

Reuse without semantic changes:

- `research_dataset.py` as the only attention input boundary;
- exact chunked RR source-set materialization in `data.py`;
- Set Transformer primitives in `set_layers.py`;
- source-group splitting, frozen-file binding, and label firewall from
  `experiment_protocol.py`;
- BF16/FP16, per-layer checkpointing, and execution chunk controls.

Retire from the active method:

- scalar-field masking and three-value member imputation;
- head/layer masked reconstruction as the primary objective;
- latent Mahalanobis as a required score;
- six-component equal Fisher fusion;
- the assumption that every hallucination must be an upper-tail prediction
  error.

`masking.py` is replaced by `corruptions.py` and should contain no active logic.

## 2. Modules and responsibilities

### `config.py`

Owns source-set, encoder, corruption, training, and calibration settings.  It
separates scientific hyperparameters from exact execution chunk sizes.

### `data.py`

Owns exact retained RR event extraction and bounded route/received-memory set
materialization.  It must not import model or label code.

### `corruptions.py`

Owns contiguous token/layer/head corruption plans and the five invariant-aware
source-set transformations.  It returns the exact channel mask and corruption
type used for self-supervision.

### `set_layers.py`

Owns permutation-equivariant/invariant Set Transformer operations only.

### `model.py`

Owns:

- typed source-member encoding;
- route and received-memory set encoders;
- head, depth-ancestry, depth-trajectory, and token-time encoders;
- online and EMA teacher encoders;
- token/channel general and type-specific energy heads.

It must not read datasets, labels, or write artifacts.

### `losses.py`

Owns robust trimmed clean-energy loss, corruption ranking, type classification,
EMA recovery, and VICReg-style variance/covariance regularization.

### `trainer.py`

Owns label-free optimization, balanced corruption scheduling, EMA updates, AMP,
checkpoint execution, and frozen row extraction.

### `calibration.py`

Owns causal condition keys and empirical tail calibration.  The primary score is
only the calibrated general energy.

### `artifacts.py`

Owns v2 checkpoint/reference/score schemas and strict validation.

### `experiment.py`

Owns fit/calibration/test source-group separation and post-hoc evaluation.

### `main.py` and `run.sh`

Own CLI and one-command smoke/full execution.

## 3. Implementation order

1. Add corruption plans and invariant tests.
2. Refactor the existing hierarchy into a reusable `SetFlowEncoder`.
3. Add online/EMA teacher wrapper and energy heads.
4. Replace V1 losses with energy, ranking, recovery, variance, and covariance.
5. Replace deterministic masked scoring and Fisher calibration with clean energy
   scoring and single-component calibration.
6. Bump artifact schemas and remove obsolete V1 fields.
7. Update smoke/full runner and project documentation.
8. Run compile, corruption invariants, exact materialization, backward, artifact,
   and end-to-end label-firewall tests.

## 4. Frozen decision rules

The general energy direction is fixed by synthetic training: corrupted source
flows have higher energy.  Type-specific energies are diagnostics and cannot be
post-hoc promoted to the primary detector.

The active implementation is rejected if any of the following holds:

- clean and corrupted energy do not separate on held-out synthetic corruptions;
- the online embedding variance remains collapsed;
- full MG-CASF does not beat the received-support causal residual baseline on an
  untouched development split;
- the result disappears under source-group bootstrap;
- the model succeeds only through total mass, edge count, or position shortcuts.

## 5. Validation matrix

Unit tests:

- every corruption preserves causality (`source < target`);
- concentration and self-reinforcement preserve row mass and membership;
- localization preserves weight multiset and member count;
- route freezing reduces route change on the selected span;
- homogenization reduces cross-head source disagreement;
- query chunking equals the dense received-support definition;
- EMA parameters receive no gradient and update by the declared momentum;
- loss is finite and backpropagates through the online encoder and energy heads;
- primary score uses only general energy;
- fit/score do not open labels.

Integration tests:

- synthetic train/calibration/test source groups are disjoint;
- one-command smoke produces model, reference, scores, and evaluation;
- all response token rows are present exactly once;
- old v1 artifacts are rejected by v2 loaders.