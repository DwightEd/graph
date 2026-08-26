# DBGNN reference on saved token graphs

This directory uses the original neural operators from *De Bruijn goes Neural:
Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs*
as a controlled reference for the saved attention-token graphs.

It produces one learned embedding for every token node. The final unsupervised
detector reads only those embeddings; it does not aggregate neighbours again.
See [METHOD.md](METHOD.md) for the exact layer-time graph construction and the
claim boundary.

## Install

From the repository environment:

```bash
pip install -r requirements.txt
pip install -r experiments/dbgnn_reference/requirements.txt
```

The copied upstream files are limited to `vendor/dbgnn.py` and `vendor/gcn.py`,
pinned in the adapter to commit `2613afe5c63183229470164f5decc2bca1a1826e`.

## Run DBGNN

The input indices must be previously saved token-graph bundles, not raw
attention directories:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
TRAIN_INDEX=/path/to/grounded_route/calibration/index.npz \
TEST_INDEX=/path/to/grounded_route/test/index.npz \
TEST_SPLIT=/path/to/RAGTruth/attention/llama31_8b/test \
OUT=experiments/dbgnn_reference/outputs/qa/dbgnn \
MODEL=dbgnn \
DEVICE=cuda \
EPOCHS=8 \
bash experiments/dbgnn_reference/run.sh
```

The cluster paths already used by this repository are in `run_qa.sh`.

## Run the necessary first-order control

Use the same command with only these values changed:

```bash
OUT=experiments/dbgnn_reference/outputs/qa/gcn \
MODEL=gcn \
bash experiments/dbgnn_reference/run.sh
```

Both runs write:

```text
checkpoint.pt          label-free encoder checkpoint
calibration/graphs/    graph sidecars with replaced node_embedding
calibration/index.npz  train-split response-node embeddings
test/graphs/           test topology plus DBGNN/GCN node embeddings
test/index.npz         frozen test response-node embeddings
detector.npz           PCA-kNN reference
scores.npz             frozen node-only anomaly scores
evaluation.json        labels opened here, after score freezing
```

## Export the compact node dataset

After the GCN run, combine the calibration/test embeddings and their binary
node labels into one shareable file:

```bash
SOURCE=experiments/dbgnn_reference/outputs/qa_compare/gcn \
TRAIN_SPLIT=/path/to/RAGTruth/attention/llama31_8b/train \
TEST_SPLIT=/path/to/RAGTruth/attention/llama31_8b/test \
OUT=experiments/dbgnn_reference/outputs/gcn_node_data_qa.npz \
bash experiments/dbgnn_reference/package_gcn_embeddings.sh
```

The output concatenates the two encoded node sets and contains only
`node_embeddings` and row-aligned binary `node_labels`. See
[GCN_NODE_EMBEDDING_BUNDLE.md](GCN_NODE_EMBEDDING_BUNDLE.md) for the array
contract.

For the actual construction test, run the first-order GCN, causal DBGNN and the
same DBGNN with only its high-order transitions removed. Labels open once,
after every detector score is frozen:

```bash
TRAIN_INDEX=/path/to/grounded_route/calibration/index.npz \
TEST_INDEX=/path/to/grounded_route/test/index.npz \
TEST_SPLIT=/path/to/RAGTruth/attention/llama31_8b/test \
BASE_OUT=experiments/dbgnn_reference/outputs/qa_compare \
DEVICE=cuda EPOCHS=8 DIAGNOSTIC_EPOCHS=20 \
bash experiments/dbgnn_reference/run_compare.sh
```

`diagnostics/report.json` contains six node-only unsupervised detectors, paired
source-bootstrap deltas for `causal - no_transition` (primary) and
`causal - GCN` (auxiliary), and source-grouped linear/MLP readability ceilings.
The supervised section is diagnostic, not the main unsupervised result.

One run is exploratory. Repeat the complete three-encoder experiment rather
than changing only the downstream detector seed:

```bash
for SEED in 20260826 20260827 20260828; do
  SEED=${SEED} BASE_OUT=experiments/dbgnn_reference/outputs/qa_compare_${SEED} \
  bash experiments/dbgnn_reference/run_compare.sh
done
```

## What is original and what is adapted

- Original: the authors' `GCN`, `HO_GCN`, high-order/first-order parallel
  message passing, terminal-node bipartite projection.
- Adapted: attention events to order-2 tensors, fixed-size input features,
  label-free endpoint training, embedding artifacts and anomaly evaluation.
- Not used: the original supervised cross-entropy training, per-graph identity
  matrices and preprocessed `.ngram` datasets.

This baseline does not imply that DBGNN must work on hallucination detection.
Its purpose is to isolate whether temporal high-order paths improve the node
representation at all.
