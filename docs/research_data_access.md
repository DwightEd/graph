# Research data access

## Metadata names

`data_source` is the original task corpus used by RAGTruth, not a model name. Examples include `CNN/DM`, `MARCO`, and `Yelp`.

`generator_model` is the model recorded in RAGTruth `response.jsonl` that generated the response.

`observer_model` is the white-box model whose attention/hidden states were extracted. It is stored once in the canonical split manifest and exposed by `ResearchSample.observer_model`.

After enrichment, every canonical `index.jsonl` row contains:

```text
sample_id
source_id
split
task_type
data_source
generator_model
temperature
quality
path
sha256
bytes
```

The attention NPZ format is unchanged.

## Enrich an already-built archive

This command only rewrites canonical `index.jsonl` / `manifest.json`. It does not touch any attention NPZ, graph PT, hidden-state NPZ, or token-stat NPZ.

If a graph cache was already built, pass `--graph-root`. Before writing either archive, enrichment verifies the graph index hash, count, sample IDs, and that its input hashes match the current canonical archive. It then updates only the graph manifest's input hashes; graph `.pt` files are not rebuilt.

```bash
python main.py enrich-index \
  --canonical-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b \
  --dataset-path /path/to/RAGTruth/dataset \
  --graph-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/graphs/llama31_8b/original_tau0p05

python main.py verify-attention \
  --archive-root /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/model_traces/llama31_8b
```

The raw dataset directory must contain `response.jsonl` and `source_info.jsonl`.

## Access in experiments

```python
from research_dataset import ResearchDataset

train = ResearchDataset(
    "/share/home/.../model_traces/llama31_8b/train",
    graph_roots={
        "original": "/share/home/.../graphs/llama31_8b/original_tau0p05/train",
    },
    device="cpu",
)

sample = train["10005"]
print(sample.metadata)
print(sample.task_type)
print(sample.data_source)
print(sample.generator_model)
print(sample.observer_model)

a = sample.attention()
g = sample.graph("original")
original = sample.original_graph()  # rebuild from canonical CSR at attention_floor
x_hidden = sample.hidden()
x_stats = sample.stats()
x = sample.node_features("attention")
edges = sample.attention_edges()
```

Filter by metadata without manually joining JSON files:

```python
qa = train.filter(task_type="QA")
cnn_dm = train.filter(data_source="CNN/DM")
```

## `positive_runs`

`labels.jsonl` is an evaluation-only sidecar. Each `positive_runs` item is a response-relative half-open token interval `[start, end)` covering consecutive hallucination-positive tokens.

For example:

```json
{"sample_id": "10005", "positive_runs": [[3, 6], [9, 10]]}
```

means response token positions `3,4,5,9` are positive. Prompt positions are not included in these coordinates.

```python
labels = train.labels()
y_response = labels.response_labels(sample)
y_full = labels.token_labels(sample)
```

`ResearchDataset` binds a graph cache to its canonical split on construction. A graph manifest must match the canonical manifest and index hashes, and its sample IDs must match exactly. Loading a graph always checks its file size, and `verify_hashes=True` additionally checks the graph SHA256. Hidden states and token statistics must carry the same `token_ids` as canonical attention. Label tensors are created on the sample attention device.

`positive_runs` preserves the binary token-level label used for evaluation. It does not preserve all original RAGTruth character-span annotation fields such as `label_type`, `meta`, or span text; those remain in the original RAGTruth `response.jsonl`.
