# Mechanism-Guided Causal Attention Set-Flow

MG-CASF is a label-free RR-attention detector.  It trains a hierarchical
source-set encoder with causal structural corruptions, an EMA teacher, and a
learned anomaly-energy head.  Hallucination labels are opened only after the
complete score artifact is frozen.

Read first:

```text
METHOD.md
IMPLEMENTATION_PLAN.md
MEMORY_AND_FIDELITY.md
```

## Pull and validate

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
git switch main
git pull --ff-only origin main

python -m unittest \
  tests.test_causal_setflow_memory \
  tests.test_causal_setflow_corruptions \
  tests.test_causal_setflow_method -v
```

## Faithful smoke run

The smoke run uses the same source-set bounds and encoder as the full run.  It
changes only sample count and epoch count.

```bash
OUT=experiments/causal_setflow/outputs/mechanism_guided/smoke_run1 \
TRAIN_LIMIT=64 \
TEST_LIMIT=5 \
EPOCHS=1 \
PRECISION=auto \
ACTIVATION_CHECKPOINTING=1 \
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash experiments/causal_setflow/run.sh
```

Expected files:

```text
model.pt
reference.npz
test_scores.npz
evaluation.json
```

One epoch over 48 fit samples remains a runtime smoke test.  It is not enough to
judge the learned energy.  Inspect the training row for:

```text
energy_gap          corrupted energy minus paired clean energy; should become positive
embedding_std       should remain clearly above zero
clean_recovery      should decrease
ranking             should decrease
```

## Full run

```bash
OUT=experiments/causal_setflow/outputs/mechanism_guided/full_run1 \
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash experiments/causal_setflow/run.sh
```

Default full training uses five epochs, BF16 when supported, exact source-set
materialization, per-layer activation checkpointing, and the complete corruption
bank.

## View component results

```bash
python - <<'PY'
import json
path = "experiments/causal_setflow/outputs/mechanism_guided/full_run1/evaluation.json"
report = json.load(open(path, encoding="utf-8"))
for name, metrics in report["components"].items():
    if metrics is None:
        continue
    print(name, f"AUROC={metrics['auroc']:.4f}", f"AUPRC={metrics['auprc']:.4f}")
PY
```

The only primary detector is:

```text
primary = causal empirical tail of general_energy
```

`type_collapse`, `type_localize`, `type_freeze`, `type_homogenize`, and
`type_self_reinforce` are predeclared diagnostics.  They must not be selected as
the final score after reading test labels.

## Memory controls

The following parameters change execution schedule only:

```text
MATERIALIZE_QUERY_CHUNK_SIZE
SET_ROW_CHUNK_SIZE
MIXER_TOKEN_CHUNK_SIZE
```

If a faithful run exceeds GPU memory, reduce those values.  Do not reduce
`HIDDEN_DIM`, source-set bounds, layer/head count, or model depth as a memory
workaround.