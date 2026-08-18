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
- `spectral_feasibility/`: the active RR causal source-persistence
  representation and independently calibrated robust-subspace detector.
- `rr_signal_audit/`: RR-only decomposition of the historical mixed coordinate
  into future received support, diagonal contribution, persistence ratio, and
  current-token collapse variables. It compares independent and joint
  one-class geometry and includes a conditional channel-shuffle control.
- `rr_topology_dynamics/`: label-free extraction followed by post-hoc analysis
  of route convergence, prompt-grounded versus self-reinforcing RR flow, and
  layer/head/source/lag attribution of spectral-subspace escape.
- `causal_attention_setwalk/`: an independent fixed SetWalk-style attention
  hypergraph representation with no-walk, pairwise, and layer-order controls.
- `causal_multiplex_flow/`: source-aware dynamic routing prediction retained as
  a research baseline; its source-prediction surprise did not establish useful
  hallucination separation.
- `conditioned_benchmark/`: a strict frozen-artifact workflow for task slices,
  token/response evaluation, uncertainty, and controlled positive prevalence.
  It never refits a detector after labels open.
