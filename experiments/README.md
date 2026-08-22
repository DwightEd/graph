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

- `graph_structure_audit/`: the prerequisite learned recoverability audit. It
  stores one exact token-pair edge tensor `[layer, head]` per sample, masks
  channels and node diagonals, and reconstructs them with explicit layer-ordered
  message passing. No `PrefixState` or handcrafted motif inventory is used in
  the main path. Message, endpoint, layer-order, head-identity, and channel-
  averaging controls determine whether graph structure actually helps.
- `source_reuse_contrast/`: the active attention-graph research line. Its former
  exact-source predictability objective is retained as a negative baseline. The
  new grounding-sensitive pipeline reconstructs high-dimensional received-
  support and prompt-origin fields, computes label-free edge sensitivity,
  refines the graph, and evaluates prompt/response counterfactual sufficiency.
- `attention_phenomenology/`: the primary mechanism audit for routing detection,
  head-coalition fracture, prompt-provenance integration, fracture-to-lock-in
  dynamics, and exact-endpoint topology. It preserves `[token, layer, head]`,
  separates known-route geometry from censoring controls, and compares real
  endpoints with a coarse-role-preserving rewired null.
- `rr_signal_audit/`: evidence-grounded decomposition of PR/RR channel fields,
  received support, collapse variables, and independent versus joint one-class
  geometry.
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
