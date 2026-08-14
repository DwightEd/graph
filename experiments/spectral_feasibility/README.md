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

## Default experiment data

The current project experiments use the existing RAGTruth-derived formal sparse
attention cache:

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/
outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

Its split roots are:

```text
TRAIN=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/train
TEST=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876/test
```

`--split-root` must point to one of these split directories containing its own
`manifest.json`; `/path/to/train` and `/path/to/test` are documentation
placeholders and must not be passed literally.

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

## Run the complete experiment

From the repository root:

```bash
bash experiments/spectral_feasibility/run.sh
```

For a quick smoke run on the first five train/test samples:

```bash
LIMIT=5 bash experiments/spectral_feasibility/run.sh
```

Override the data root or output only when needed:

```bash
ROOT=/another/formal/cache \
OUT=experiments/spectral_feasibility/outputs/custom \
DEVICE=cpu \
bash experiments/spectral_feasibility/run.sh
```

The runner performs, in order: label-blind train extraction, label-blind test
extraction, unlabeled robust-reference fitting/test scoring, and post-hoc label
evaluation.

## Run individual stages

```bash
ROOT=/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876

python -m experiments.spectral_feasibility.main extract \
  --split-root "$ROOT/train" \
  --output outputs/spectral/train_features.npz \
  --device cpu

python -m experiments.spectral_feasibility.main extract \
  --split-root "$ROOT/test" \
  --output outputs/spectral/test_features.npz \
  --device cpu

python -m experiments.spectral_feasibility.main score \
  --train-features outputs/spectral/train_features.npz \
  --test-features outputs/spectral/test_features.npz \
  --output outputs/spectral/test_scores.npz

python -m experiments.spectral_feasibility.main evaluate \
  --split-root "$ROOT/test" \
  --scores outputs/spectral/test_scores.npz \
  --output outputs/spectral/evaluation.json
```

For a single-sample smoke test:

```bash
python -m experiments.spectral_feasibility.main extract \
  --split-root "$ROOT/test" \
  --sample-id 10071 \
  --output outputs/spectral/sample_10071.npz
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
