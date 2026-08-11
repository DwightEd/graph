# Graph-only t-SNE

Use `scripts/graph_tsne.py` when the goal is to project the reconstructed sparse graph dataset itself. Unlike `notebooks/graph_tsne.ipynb`, this path does not reload canonical attention features and does not require `ATTENTION_ROOT` for the embedding.

## Required graph split layout

`--graph-root` must point to one split directory, not to the parent model directory:

```text
<graph-root>/
├── manifest.json
├── index.jsonl
└── graphs/
    ├── <sample_id>.pt
    └── ...
```

For the default reconstruction script this is:

```text
/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test
```

The `graphs/` child is created by `main.py build`. If it does not exist, the directory is either not a graph split or the graph build has not been run there.

To locate reconstructed graph splits on the server:

```bash
find /share/home/tm902089733300000/a903202310/lys/data/RAGTruth \
  -type f -name manifest.json -print
```

A valid graph manifest has `"schema": "ragtruth-token-graph-v1"` and a `kind` such as `relation_topk_channels`.

## Run directly from reconstructed `.pt` graphs

Without labels:

```bash
python scripts/graph_tsne.py \
  --graph-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test \
  --output-dir outputs/graph_tsne/test
```

With correct/hallucinated colors, pass the canonical label sidecar separately:

```bash
python scripts/graph_tsne.py \
  --graph-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test \
  --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test/labels.jsonl \
  --output-dir outputs/graph_tsne/test
```

The label file is read only after graph descriptors and t-SNE coordinates are computed. It does not affect feature extraction, scaling, PCA, or t-SNE fitting.

## What is projected

For every saved graph, `token_behavior_features` computes 11 response-token graph features:

- incoming mass
- prompt mass share
- normalized routing entropy
- response-history lag
- total incoming degree
- prompt degree
- response-history degree
- incoming density
- prompt density
- response-history density
- response-history edge share

Each variable-length response graph is summarized with mean, population standard deviation, and linear slope over response position:

```text
11 token graph features × (mean, std, slope) = 33-D graph descriptor
```

The pipeline is therefore:

```text
saved sparse graph .pt
    -> token-level graph behavior [R, 11]
    -> mean/std/slope pooling [33]
    -> StandardScaler
    -> PCA only if dimension > 50
    -> t-SNE [2]
```

Outputs:

- `graph_tsne.png`: scatter plot
- `graph_tsne_coordinates.npz`: sample IDs, 33-D descriptors, descriptor names, t-SNE coordinates, and optional labels
- `metadata.json`: graph kind, sample count, dimensions, and run configuration

## If the graph split has not been built

The canonical attention archive is not itself the graph dataset. Build the graph split once:

```bash
python main.py build \
  --cache-dir /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b/test \
  --output-dir /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/relation_topk_channels/test \
  --kind relation_topk_channels \
  --k-prompt 8 \
  --k-history 8 \
  --device cuda
```

This command creates the `manifest.json`, `index.jsonl`, and `graphs/*.pt` contract expected by graph-only t-SNE.
