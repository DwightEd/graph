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

## Resume and CPU parallelism

Every sample is written atomically.  The runner now validates existing sample
artifacts, skips valid ones, and maintains ``run_state.json`` plus an atomic
``index.jsonl`` checkpoint.  To continue an interrupted run, reuse its exact
top-level output directory and the same rank/seed/diagonal settings:

```bash
OUTPUT_ROOT=/path/to/existing/run \
RESUME=1 WORKERS=2 RANK=16 \
bash attention_multiplex/run_attention_multiplex.sh
```

The two sparse randomized SVDs are CPU operations; setting
``CUDA_VISIBLE_DEVICES`` does not accelerate them. ``WORKERS`` runs different
samples concurrently. Start with 2, and use 4 only when RAM and CPU headroom
are available. The script pins the native BLAS/OpenMP pools to one thread to
avoid oversubscription.

Artifacts from the immediately preceding non-checkpointing version can also
be adopted with ``RESUME=1``. This is safe only when the resumed command uses
the same ``RANK``, ``SEED``, and ``INCLUDE_DIAGONAL`` values as the interrupted
command; those legacy files predate embedded configuration metadata.

The progress bar advances once per sample. Each split contains:

```text
manifest.json
run_state.json
index.jsonl
samples/multiplex_<sample_id>.npz
samples/multiplex_<sample_id>.json
```

See [METHOD.md](METHOD.md) for the exact graph and matrix definitions.
