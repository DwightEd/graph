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
reimplement canonical CSR parsing. Shared format-independent views belong in
`research_dataset.py`; reusable graph/event construction belongs in
`attention_graph/`.

Allowed experiment-local I/O is limited to artifacts produced by the experiment
itself, such as derived embeddings, scores, figures and reports.

## Label discipline

Representation construction, parameter fitting, clustering, density fitting
and anomaly scoring must remain label-blind unless a directory explicitly
describes a supervised diagnostic. Evaluation labels open only after frozen
score artifacts exist.

## Current experiments

- `mechanism_validation/`: attention mechanism and graph-ablation diagnostics.
- `spectral_feasibility/`: RR causal spectral representation and independently
  calibrated robust-subspace detector.
- `rr_topology_dynamics/`: post-hoc mechanism audit of route convergence,
  source/lag dynamics and spectral residual attribution.
- `causal_isomorphism_trajectory/`: non-neural, condition-aware PPCA geometry
  over temporal-WL-inspired causal event signatures, generation transitions
  and ordered layer-depth transitions.
- `conditioned_benchmark/`: strict comparison of frozen detector artifacts under
  aligned task, token/response and prevalence conditions.

The retired `causal_multiplex_flow/` neural source-prediction detector is
preserved only as a strict historical artifact reader. Its negative result is
documented in `docs/results/cmrp_negative_result.md`.
