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

## Active evidence path

- `mechanism_validation/`: attention mechanism and graph-ablation diagnostics.
- `spectral_feasibility/`: the reproduced RR causal source-persistence
  representation and independently calibrated robust-subspace detector.
- `rr_signal_audit/`: the active evidence-grounded attention audit. It preserves
  PR/RR role fields per layer/head, decomposes received support, reproduces the
  earlier exact scalar baselines, and compares independent with joint one-class
  geometry under a conditional channel-shuffle control.
- `rr_topology_dynamics/`: label-free extraction followed by post-hoc analysis
  of route convergence, prompt-grounded versus self-reinforcing RR flow, and
  layer/head/source/lag attribution of spectral-subspace escape.
- `conditioned_benchmark/`: a strict frozen-artifact workflow for task slices,
  token/response evaluation, uncertainty, and controlled positive prevalence.
  It never refits a detector after labels open.

## Retired negative controls

- `causal_attention_setwalk/`: fixed hypergraph walk. Its smoke result was worse
  than the no-walk control, so it is retained only as a falsified mechanism.
- `causal_multiplex_flow/`: source-prediction surprise did not establish useful
  separation and is not an active representation.

The incomplete neural `causal_setflow` prototype was removed. It mixed a custom
Set Transformer, EMA teacher, and synthetic corruption bank, excluded all PR
edges, lacked its documented decoder/ablations, and its entry point no longer
matched its calibration/trainer interfaces. It must not be cited as a working
method or empirical result.
