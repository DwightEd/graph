# Spectral feasibility

This directory tests whether node-local spectral geometry of the retained
response-attention operator provides a stronger unsupervised token
representation than scalar graph statistics.

## Data boundary

Raw attention is never opened here. Every command starts with
`research_dataset.open_research_dataset()`. Sparse CSR decoding and the
channel-mean `[R,N]` response-attention view are provided by
`research_dataset.py`.

The spectral code therefore works with both canonical NPZ splits and formal PT
caches through the same `ResearchSample` API.

## Representation

For each response token we construct relation-specific transport matrices from
the channel-mean retained attention:

```text
A_RP : response queries -> prompt sources
A_RR : response queries -> previous response sources
```

The first feasibility representation concatenates:

1. **RP local HKS** on the response-response co-attention graph
   `A_RP A_RP^T`;
2. **RR local HKS** on `A_RR A_RR^T`;
3. sign-invariant **RP receiver SVD band energy**;
4. sign-invariant **RR receiver SVD band energy**;
5. sign-invariant **RR sender SVD band energy**.

The representation is constructed without `positive_runs` or `y_token`.
Unretained values are zero-filled only in the requested dense view and still
mean `<= attention_floor`, not known original zeros.

## Run

Extract train and test representations:

```bash
python -m experiments.spectral_feasibility.main extract \
  --split-root /path/to/train \
  --output outputs/spectral/train_features.npz \
  --device cpu

python -m experiments.spectral_feasibility.main extract \
  --split-root /path/to/test \
  --output outputs/spectral/test_features.npz \
  --device cpu
```

For a single sample smoke test:

```bash
python -m experiments.spectral_feasibility.main extract \
  --split-root /path/to/test \
  --sample-id 10071 \
  --output outputs/spectral/sample_10071.npz
```

Fit a completely unlabeled trimmed robust Gaussian reference on train vectors
and score test vectors by Mahalanobis distance:

```bash
python -m experiments.spectral_feasibility.main score \
  --train-features outputs/spectral/train_features.npz \
  --test-features outputs/spectral/test_features.npz \
  --output outputs/spectral/test_scores.npz
```

Only after the scores are frozen, open evaluation labels:

```bash
python -m experiments.spectral_feasibility.main evaluate \
  --split-root /path/to/test \
  --scores outputs/spectral/test_scores.npz \
  --output outputs/spectral/evaluation.json
```

The evaluation reports overall anomaly-score AUROC/AUPRC and the post-hoc
separability of each individual spectral coordinate. Individual-coordinate
AUROC is diagnostic only; it never feeds back into representation or scoring.

## Interpretation

This is a feasibility experiment, not the final method. A positive result means
that RP/RR spectral node signatures preserve hallucination-relevant structure
that scalar statistics or raw node embeddings may discard. The next stage can
then replace the fixed channel mean with learned channel mixtures and add
causal spectral innovation across response steps.
