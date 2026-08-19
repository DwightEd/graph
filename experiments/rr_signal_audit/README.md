# Evidence-grounded causal attention signal audit

This is the active attention-only mechanism experiment. It preserves
layer-by-head fields and tests three separately attributable families:

- exact PR/RR route marginals, including the previously useful history-edge
  fraction and prompt/history per-source mass;
- RR received-support persistence, whose strict causal residual was the
  strongest reproduced structural signal;
- local RR concentration and route dynamics as an explicit hypothesis block.

There is no GNN, synthetic-anomaly learner, or backpropagation. Robust
standardization/PCA is fitted without labels and is reported beside independent
coordinate density and direct historically frozen scalar baselines.

See `EVIDENCE.md` for the reproduced observations and mandatory acceptance
gates. A new propagation or fusion method is not promoted unless it clears
those gates.

## Smoke test

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

TRAIN_LIMIT=128 \
TEST_LIMIT=5 \
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash experiments/rr_signal_audit/run.sh
```

Default smoke output:

```text
experiments/rr_signal_audit/outputs/smoke_train128_test5/
├── reference.npz
├── test_scores.npz
└── evaluation/
    ├── evaluation.json
    ├── score_metrics.csv
    └── onset_effects.csv
```

Use a fresh output directory for a rerun:

```bash
OUT=experiments/rr_signal_audit/outputs/smoke_retry \
TRAIN_LIMIT=128 TEST_LIMIT=5 \
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
bash experiments/rr_signal_audit/run.sh
```

## Full audit

```bash
CUDA_VISIBLE_DEVICES=0 \
DEVICE=cuda \
bash experiments/rr_signal_audit/run.sh
```

## What to inspect first

In `score_metrics.csv`, first compare the same conditioning/model across:

```text
mixed_topk
received_topk
diagonal_topk
ratio_topk
collapse_channel
prompt_route_channels
history_route_channels
history_edge_channels
edge_strength_channels
```

Interpretation:

- `received_topk > diagonal_topk`: subsequent RR source use is more informative
  than diagonal self-attention.
- `diagonal_topk >= mixed_topk`: the historical result is largely a diagonal
  artifact.
- `ratio_topk > received_topk`: relative persistence is more informative than
  absolute received support.
- `ppca_nll_tail > independent_nll_tail`: joint channel dependence may add
  information beyond marginal shifts.
- `causal` close to `relative`: the result does not depend strongly on final
  response length.
- `relative` much stronger than `causal`: historical performance may rely on
  offline final-length conditioning.

Rows with `family=historically_frozen_scalar_baseline` reproduce five earlier
audited scalar definitions unchanged. They are controls, not the new node
representation. The four `*_channels` blocks retain all layer/head coordinates
instead of averaging them into those scalars.

In `evaluation.json`, inspect `coordination`. A passed channel-shuffle gate only
shows that channel alignment is learnable; it must not be described as a
correctness mechanism unless its frozen score also separates hallucination.

In `onset_effects.csv`, the local-collapse hypothesis predicts:

```text
source_entropy                  negative
log_source_effective_number     negative
source_top1_share               positive
log_source_mean_lag             negative
source_local_mass_share         positive
anchor_turnover                 negative
lag_route_velocity              negative
route_effective_rank            negative
```

## Label discipline

`fit` and `score` never open hallucination labels. `evaluate` opens labels only
after `test_scores.npz` is frozen. Fit, calibration, and test are separated by
complete `source_id` groups.

See `METHOD.md` for formulas, claim boundaries, and decision rules.
