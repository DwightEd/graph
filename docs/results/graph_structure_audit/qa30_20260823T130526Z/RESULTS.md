# Graph structure audit: QA30 smoke result

This directory publishes the evaluation artifacts for the 30-sample QA smoke
run completed on 2026-08-23. It is an exploratory structure audit, not a final
hallucination detector result.

## Provenance

- Implementation commit: `dcebc096e883e3a8868e9e85ab0047a25473e674`
- Dataset: RAGTruth, Llama-3.1-8B attention cache, QA task
- Fit split: 27 train samples
- Validation split: 3 train samples
- Scored split: 30 test samples
- Scored tokens: 6,866
- Hallucinated tokens opened at evaluation: 415 (6.04%)
- Hidden width: 16
- Epochs: 2
- Score rounds: 2
- Sparse decode block rows: 4,096
- Seed: 20260822

Hallucination labels were not read during fitting or score generation. They were
opened only by the post-hoc evaluator.

## Headline results

- Recovery AUROC: 0.6082
- Recovery AUPRC: 0.1405, compared with 0.0604 prevalence
- Hallucination minus correct recovery loss: +0.002800
- Matched hallucination minus correct recovery loss: +0.002425
- Message gain: +0.00002755; statistically stable but only about 0.13% of the
  mean recovery loss
- Layer-order gain: +0.00023143
- Head-identity gain: +0.00001675
- Endpoint gain: +0.00000433
- Layer-head gain: -0.00091939
- Full-channel gain: -0.00151432

The audit's required fine-grained representation gates did not all pass:
layer-head and full-channel inputs were worse than their collapsed controls.
Accordingly, this run does not support using the learned graph embedding as a
final unsupervised hallucination representation.

The positive message gate only establishes a small masked-reconstruction
benefit for the audit model's neighbor aggregation. It does not establish
evidence grounding or a causal effect on the base LLM.

## Published artifacts

- `evaluation.json`: machine-readable audit conclusions and structure gates
- `metrics.csv`: token-level discrimination metrics
- `matched_effects.csv`: position/density-matched effects
- `recoverability.csv`: direction of recovery differences
- `structure_gates.csv`: structural ablation gates and source-bootstrap intervals

The local `model.pt` and per-token `scores.npz` are intentionally excluded
from this presentation bundle. The original run manifests also contained local
absolute paths; this report records the portable protocol instead.

## Reproduction profile

```bash
ROOT=/path/to/RAGTruth/llama31_8b TASK_TYPE=QA \
TRAIN_LIMIT=30 TEST_LIMIT=30 EPOCHS=2 SCORE_ROUNDS=2 \
HIDDEN_DIM=16 BLOCK_ROWS=4096 DEVICE=cpu SAVE_GRAPHS=0 \
  bash experiments/graph_structure_audit/run.sh
```

## Limitations

This is a one-seed, deliberately under-trained smoke run. It uses retained
attention traces and a frozen-model perturbation audit; it does not separately
retrain matched-capacity ablations. Strong method claims require a larger,
multi-seed run and explicit grounding-lineage experiments.
