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

## Mandatory flat all-layer baseline

`flat-1024` keeps the same exact token-pair `layer x head` tensor but removes all
adjacency. For a 32-layer, 32-head model, each pair has 1024 raw attention
coordinates. It uses the same source split, block masking, conditional density
and evaluation protocol.

```bash
TRAIN_SPLIT=/path/to/train \
TEST_SPLIT=/path/to/test \
OUT=experiments/holoroute/outputs/flat_1024 \
TASK_TYPE=QA DEVICE=cuda \
bash experiments/holoroute/run_flat1024.sh
```

Run both with identical environment variables:

```bash
TRAIN_SPLIT=/path/to/train TEST_SPLIT=/path/to/test \
OUT_ROOT=experiments/holoroute/outputs/comparison \
TASK_TYPE=QA DEVICE=cuda \
bash experiments/holoroute/run_comparison.sh
```

See `FLAT1024_BASELINE.md` for the exact control design. The first implementation
intentionally excludes CUSUM, Markov state tables, prompt-anchor claims and
spectral hybrids. See `METHOD.md` and `EXPERIMENT_PLAN.md` before interpreting
results.
