# RR signal decomposition audit

This experiment uses **RR attention only** to determine what produced the
historical residual signal and whether hallucination onset exhibits local
routing collapse. It is intentionally an audit before any new neural model.

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
experiments/rr_signal_audit/outputs/v1/smoke_train128_test5/
├── reference.npz
├── test_scores.npz
└── evaluation/
    ├── evaluation.json
    ├── score_metrics.csv
    └── onset_effects.csv
```

Use a fresh output directory for a rerun:

```bash
OUT=experiments/rr_signal_audit/outputs/v1/smoke_retry \
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

In `score_metrics.csv`, compare the same conditioning/model across:

```text
mixed_topk
received_topk
diagonal_topk
ratio_topk
collapse_channel
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
