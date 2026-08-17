# Causal Isomorphism Trajectory Geometry

CITG is a label-free, non-neural attention-graph detector. It represents each
response token with a bounded temporal-isomorphism invariant, generation-time
changes and ordered layer-depth changes, then scores the trajectory under a
task/causal-phase-conditioned PPCA reference.

The implemented signature is a deterministic two-hop temporal-WL-style
invariant, not a complete graph-isomorphism solver. The primary score is fixed
before evaluation labels are opened.

Read:

1. `METHOD.md` for the scientific design;
2. `IMPLEMENTATION_PLAN.md` for module and validation contracts;
3. this file for commands and outputs.

## What changed from CMRP

CMRP's source-routing surprise was near random on the full test set. Its generic
event extraction and lag-preserving rewire ideas were retained as shared
infrastructure; the GRU/source-prediction detector was removed.

CITG does not use epochs or backpropagation.

## Smoke test

Use enough train groups for separate fit/calibration streams:

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

TRAIN_LIMIT=128 TEST_LIMIT=5 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_isomorphism_trajectory/run.sh
```

Output:

```text
experiments/causal_isomorphism_trajectory/outputs/v1/
smoke_train128_test5/
├── reference.npz
├── test_scores.npz
└── evaluation.json
```

Smoke results validate runtime only.

## Full run

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/causal_isomorphism_trajectory/run.sh
```

Default output:

```text
experiments/causal_isomorphism_trajectory/outputs/v1/full/
```

Common fixed overrides:

```bash
LAYER_BANDS=8 \
MAX_RP_EVENTS_PER_BAND=2 \
MAX_RR_EVENTS_PER_BAND=4 \
HASH_DIM=128 \
PCA_DIM=32 \
REFERENCE_PER_SAMPLE=16 \
OUT=/absolute/fresh/output \
bash experiments/causal_isomorphism_trajectory/run.sh
```

Changing these after inspecting final test labels requires a new held-out
evaluation split.

## Reference outputs

`reference.npz` contains no labels. Important fields:

```text
fit_group_id
calibration_group_id
calibration_energy_full
calibration_energy_static
calibration_energy_topology
calibration_energy_mass

topology_gate_mean_gap
topology_gate_ci_low
topology_gate_ci_high
topology_gate_coverage
topology_gate_pass
```

A useful topology representation should give:

```text
rewired PPCA energy - true PPCA energy > 0
```

with a positive source-bootstrap lower confidence bound. Synthetic contract
tests verify that the gate is recorded; they do not force an arbitrary tiny
fixture to pass it.

## Score outputs

`test_scores.npz` contains:

```text
score                 # primary full trajectory
score_static
score_topology
score_mass

energy_full
energy_static
energy_topology
energy_mass

rewired_energy_full
rewire_energy_gap
rewire_valid
```

Only the frozen primary score is the method result. The other score fields are
preregistered ablations, not post-hoc alternatives.

## Interpretation gates

Do not report CITG as successful unless:

1. the calibration topology gate passes;
2. `score` beats `score_mass` and the current RR spectral baseline on identical
   token rows;
3. `score` beats `score_static`;
4. no score direction is inverted from test labels.

The conditioned benchmark can compare frozen artifacts after the full run.
The CI suite checks compilation, shell syntax, data boundaries, source-group
protocols, benchmark registration, and the synthetic fit-score-evaluate path.
