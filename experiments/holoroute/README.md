# HoloRoute

Unsupervised neural learning on the causal attention event graph audited in
`experiments/attention_holonomy_audit/`.

```bash
TRAIN_SPLIT=/path/to/train \
TEST_SPLIT=/path/to/test \
OUT=experiments/holoroute/outputs/qa \
TASK_TYPE=QA DEVICE=cuda \
bash experiments/holoroute/run.sh
```

The workflow performs:

```text
train neural graph encoder
-> fit position-conditioned density
-> freeze test token scores
-> open labels only in evaluate
```

The first implementation intentionally excludes CUSUM, Markov state tables,
prompt-anchor claims and spectral hybrids. See `METHOD.md` and
`EXPERIMENT_PLAN.md` before interpreting results.
