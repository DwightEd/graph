# Attention Multiplex

This is the attention-only graph-construction subproject. It consumes the
existing formal sparse cache directly through the repository's canonical
`ResearchDataset` interface.

It outputs, for every sample:

- layer-specific response-query node roles;
- head-specific prompt/response-source node roles;
- mass and square-root-distribution spectral views kept separately;
- singular values and captured energy;
- exact self attention, retained row mass, and unresolved row mass;
- no labels, scores, AUROC, t-SNE, or selected “best” sample.

The PP block is not present in the cache and is not invented. Legal censored
PR/RR edges are represented as `attention_floor` in the central reconstructed
view; the sparse spectral factorization removes that deterministic floor
baseline before SVD.

## Run

From the repository root:

```bash
bash attention_multiplex/run_attention_multiplex.sh
```

The defaults use the existing formal cache and process both `train` and
`test`. A different cache/output root can be supplied:

```bash
bash attention_multiplex/run_attention_multiplex.sh \
  /path/to/formal_attention_cache \
  /path/to/output
```

Fast interface check:

```bash
LIMIT=5 RANK=8 bash attention_multiplex/run_attention_multiplex.sh
```

The progress bar advances once per sample. Each split contains:

```text
manifest.json
index.jsonl
samples/multiplex_<sample_id>.npz
```

See [METHOD.md](METHOD.md) for the exact graph and matrix definitions.
