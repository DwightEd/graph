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

## Which token features carry label separation?

The manifests only prove extraction completeness. They do not contain token
labels or feature-separation results. After both splits finish, run the
post-hoc signal audit on the saved roles:

```bash
bash attention_multiplex/run_signal_audit.sh \
  /path/to/attention_multiplex/run
```

The audit first freezes rotation-invariant features from all train/test NPZ
files. Only then does it open the canonical evaluation labels. It reports each
feature independently; it does not fit a combined detector. Candidate families
are query leverage, layer velocity/acceleration, prompt/history route strength
per available source, head disagreement, self routing, and unresolved mass.
The prompt/history comparison divides by the respective source-token counts.

Train labels determine the exploratory direction and nonredundant shortlist;
test labels evaluate that frozen direction. A train-only robust position-bin
reference is reported alongside raw metrics to expose positional confounding.
Outputs are:

```text
signal_audit/feature_signal_report.json
signal_audit/feature_signal_ranking.csv
signal_audit/feature_signal_ranking.png
signal_audit/position_reference.npz
```

Because train labels are used for mechanism discovery, this audit identifies
promising signals but is not itself the final unsupervised detector result.
