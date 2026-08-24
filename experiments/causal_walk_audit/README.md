# Typed Route-Grammar Audit

This directory is the integrated replacement for the earlier Ridge causal-walk
prototype. It selectively absorbs the useful construction ideas from commit
`825cbec` and removes the parts that were not supported by the previous results.

The retained core is:

```text
mass-conserving exact attention graph
    -> typed layer-unfolded lineage automaton
    -> full-soft order-1/order-2 De Bruijn backoff grammar
    -> grammar rupture as the primary unsupervised score
    -> response closure as a separate mechanism diagnostic
    -> hierarchical layer/head calibration
```

## What was removed

The previous implementation mixed anchor identity, one-hop lineage and
higher-order paths inside one Ridge feature block. It also relied on prompt
chunks as evidence anchors and built `anchor_js`, `evidence_escape`,
`recoupling_failure`, and a hand-composed `lock_in` score. Their real-cache
results were weak or directionally inconsistent, so they are no longer part of
the main path.

The `825cbec` implementation also contained a large artifact framework,
spectral hybrid bridge, visualization stack, fixed top-2 soft-state truncation,
and a rupture-times-lock-in primary detector. Those pieces are not copied into
this directory. The path-only mechanism must stand on its own.

## Primary detector

For every layer/head channel, the method fits a label-free route grammar over
response-token time. A variable-order backoff model uses order two only when its
context has enough support. Predictive cross-entropy is robustly standardized
and accumulated by a causal CUSUM. The resulting per-channel rupture signals
are calibrated without labels and fused hierarchically:

```text
heads within each layer -> layers within the model -> one token score
```

The frozen primary detector is named `score`.

Response-closed persistence is still saved as `closure_mean` and
`rupture_closure_mean`, but it is not forced into the primary detector before
the mechanism is validated.

## Run

```bash
DATA_ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/causal_walk_audit/outputs/qa \
DEVICE=cuda \
TASK_TYPE=QA \
bash experiments/causal_walk_audit/run.sh
```

Small smoke:

```bash
TRAIN_LIMIT=100 TEST_LIMIT=30 DEVICE=cpu \
DATA_ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/causal_walk_audit/outputs/qa30 \
bash experiments/causal_walk_audit/run.sh
```

Outputs:

```text
reference.npz
test_scores.npz
evaluation/evaluation.json
evaluation/diagnostic_metrics.csv
```

`fit` and `score` never request hallucination labels. `evaluate` opens labels
only after the score artifact is frozen.

## Claim boundary

The method models an attention-lineage proxy. It does not observe value/output
projections, residual mixing, MLP updates, or prompt-query rows. A high score
means the observed route state violates the unlabeled generation-time grammar;
it is not by itself a causal explanation of hallucination.
