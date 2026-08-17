# CMRP negative result

## Purpose

This file preserves the scientific outcome of the retired
`experiments/causal_multiplex_flow/` detector. Reusable causal-event extraction
and lag-preserving rewiring were generalized into `attention_graph/`; the neural
source-prediction method itself was removed.

## Reported full test result

The supplied full-run evaluation reported:

| Component | AUROC | AUPRC |
|---|---:|---:|
| calibrated causal route surprise | 0.4832 | 0.0579 |
| raw route surprise | 0.4832 | 0.0579 |
| source NLL | 0.4832 | 0.0579 |
| presence NLL | 0.3988 | 0.0473 |
| rewired source NLL | 0.4392 | 0.0524 |
| rewire gap | 0.4760 | 0.0587 |
| weight error | 0.5160 | 0.0659 |

The test prevalence in the corresponding RAGTruth token split was about
0.0621. Thus the primary score was below chance in both AUROC and AUPRC.

## What the result establishes

CMRP's label-free topology gate could prefer true RR sources over
lag-preserving alternatives. This establishes that exact source identity is
learnable beyond a coarse lag shortcut.

It does **not** establish that source-prediction surprise is a correctness
signal. The full result falsifies the preregistered one-sided hypothesis:

```text
harder-to-predict RR routing -> more likely hallucination
```

The result is not repaired by reporting `1 - AUROC`, reversing a component, or
choosing `weight_error` after test labels are inspected.

## Methodological lesson

A self-supervised pretext can learn genuine topology while remaining
misaligned with factual correctness. Errors may be common, fluent,
self-reinforcing or prematurely converged rather than structurally
unpredictable.

The replacement CITG experiment therefore models conditioned trajectory
density over state and transitions, including under-motion/collapse, rather
than using source-prediction loss as the anomaly score.
