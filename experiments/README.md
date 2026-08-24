# Experiments

`experiments/` contains feasibility-stage research code. Algorithms may change;
the data boundary and label discipline must not.

## Required data contract

Every attention experiment opens data through:

```python
from research_dataset import open_research_dataset

dataset = open_research_dataset(split_root, device="cpu")
sample = dataset[sample_id]
```

Experiments must not parse canonical PT/NPZ caches directly or duplicate CSR
decoding. Shared format-independent views belong in `research_dataset.py`.
Experiment-local I/O is limited to derived references, embeddings, scores,
figures, and reports.

## Label discipline

Representation construction, reference fitting, clustering, density estimation,
and anomaly scoring remain label-blind unless a directory is explicitly marked
as a supervised diagnostic. `dataset.labels()` or
`dataset.prepare_evaluation_labels()` is reserved for a separate post-hoc stage
after representations and scores are frozen.

## Active evidence path

- `holoroute/`: proposed neural method. It reuses the audited dual-axis attention
  event graph, encodes complete head profiles, transports messages separately
  over depth and causal-relay relations, mixes query source sets, and learns by
  whole-event, path, depth, query and holonomy self-supervision. A disjoint
  unlabeled calibration stream removes position/length/coverage nuisance and
  scores the local mechanism residual vector. No CUSUM or cumulative token score
  is used.
- `attention_holonomy_audit/`: mechanism gate and graph-construction audit for
  HoloRoute. Small train-only probes test depth, relay, query-set, middle-token
  and holonomy relations before strong graph claims are made.
- `causal_walk_audit/`: statistical typed route-grammar baseline retained for
  comparison. Its earlier cumulative rupture score was strongly confounded with
  token position, so it is not the proposed main method.
- `non_neural_structure_audit/`: prompt-connected/response-base routing-lineage
  model-selection audit.
- `graph_structure_audit/`: learned masked recoverability baseline over exact
  token-pair `[layer, head]` edge tensors.
- `source_reuse_contrast/`: grounding-sensitive reconstruction and
  counterfactual-sufficiency experiments.
- `attention_phenomenology/`: routing, head-coalition, prompt-provenance,
  fracture-to-lock-in and endpoint-topology audits.
- `rr_signal_audit/`: PR/RR fields, received support, collapse variables and
  independent versus joint one-class geometry.
- `rr_topology_dynamics/`: route convergence, prompt-grounded versus unsupported
  RR relay, and layer/head/source/lag attribution.
- `mechanism_validation/`: supervised post-hoc mechanism and graph-ablation
  diagnostics.
- `spectral_feasibility/`: reproduced RR source-persistence coordinates and a
  robust-subspace baseline; it is not a graph Laplacian method.
- `conditioned_benchmark/`: frozen-artifact task/token/response evaluation under
  controlled conditions and prevalence.

## Retired negative controls

- `causal_attention_setwalk/`: fixed hypergraph walk; its smoke result was worse
  than the no-walk control.
- `causal_multiplex_flow/`: source-prediction surprise did not establish useful
  hallucination separation.

The removed neural `causal_setflow` prototype must not be cited as a working
method or empirical result.
