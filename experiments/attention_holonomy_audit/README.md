# Attention Holonomy Audit

This directory validates the structural assumptions behind the proposed
attention-only HoloRoute model. It builds a dual-axis causal event graph and
fits small train-only transport probes. It does not train the final neural graph
network and never uses hallucination labels during `fit` or `score`.

## Run

```bash
DATA_ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/attention_holonomy_audit/outputs/qa30 \
TRAIN_LIMIT=100 TEST_LIMIT=30 DEVICE=cpu \
bash experiments/attention_holonomy_audit/run.sh
```

Outputs:

```text
reference.npz                 transport maps, nuisance reference, structure gates
scores.npz                    frozen token scores and response-token IDs
maps/*.npz                    per-sample token x layer mechanism maps
evaluation/metrics.csv        same-token and shifted AUROC/AUPRC
evaluation/position_correlations.csv
evaluation/matched_effects.csv
evaluation/evaluation.json
```

Read `METHOD.md` and `EXPERIMENT_PLAN.md` before interpreting a result.
