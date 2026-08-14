# Experiments

`experiments/` contains disposable or feasibility-stage research code. The
algorithms may change quickly; the data boundary must not.

## Required data contract

Every experiment that consumes model attention must open data through:

```python
from research_dataset import open_research_dataset

dataset = open_research_dataset(split_root, device="cpu")
sample = dataset[sample_id]
attention = sample.attention()
```

Experiments must **not** import raw loaders from `cache.py` or
`formal_cache.py`, scan `attention_*.pt` / `attention/*.npz` themselves, or
reimplement canonical CSR parsing. Shared sparse/dense views belong in
`research_dataset.py` and should be added there when an experiment needs a new
format-independent data view.

Allowed experiment-local I/O is limited to artifacts produced by the
experiment itself, such as derived embeddings, scores, figures, and JSON/NPZ
reports.

## Label discipline

Representation construction, parameter fitting, clustering, density fitting,
and anomaly scoring must remain label-blind unless a directory explicitly
describes a supervised diagnostic. `dataset.labels()` is reserved for a
separate post-hoc evaluation stage after representations/scores are frozen.

## Current experiments

- `mechanism_validation/`: attention mechanism and graph-ablation diagnostics.
- `spectral_feasibility/`: node-local spectral representations derived from
  RP/RR response-attention transport matrices.
