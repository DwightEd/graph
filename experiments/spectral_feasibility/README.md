# Causal spectral subspace anomaly detection

This experiment is a fully label-free token detector for cached LLM attention.
It does **not** average layer/head channels before graph construction and does
not use hallucination labels during fitting or scoring. Labels are opened only
by the final `evaluate` command.

## Data boundary

All raw attention access goes through `research_dataset.open_research_dataset()`
and `ResearchSample.iter_sparse_attention_blocks()`. The cache contains response
query rows, exact stored diagonals, and retained off-diagonal values above the
attention floor. Missing entries remain censored; the method never claims they
are original zeros.

## 1. Token-aligned causal graph state

For every response token `t` and every channel `c=(layer, head)`, RR edges form
a causal response-prefix graph. Following the LapEigvals weighted-degree
convention,

```text
d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
lambda[c,t,j] = d[c,t,j] - A_c[j,j]
```

The causal Laplacian is triangular, so its diagonal is its spectrum. The state
keeps the `top_k` eigenvalues with largest **absolute** magnitude per channel
while preserving sign. This retains strong positive and negative departures.

Prompt query rows are unavailable, so RP does not fabricate a prompt Laplacian.
Instead each retained response-to-prompt edge is accumulated into fixed
relative prompt-position bins independently per layer/head. The sum of those
bins is exactly the retained prompt mass and the bin pattern preserves coarse
source location without CountSketch collisions.

With the default 32x32 geometry, `top_k=5` and `prompt_bins=8`, the raw token
state has

```text
1024 * (5 + 8) = 13312 dimensions.
```

## 2. Robust train-only spectral subspace

Approximately six deterministic reference tokens are taken from each train
response. Raw states are robustly standardized within four relative-position
bins using median/MAD. A first PCA estimates a provisional subspace; within each
position bin the largest 10% reconstruction-residual tail is removed, and PCA
is fitted again on the remaining unlabeled references.

```text
z_t = robust_standardize(s_t)
e_t = whiten(PCA(z_t))
r_static(t) = mean((z_t - reconstruct(z_t))^2)
```

This is a robust one-class linear reconstruction model. No train label is read.

## 3. Causal dynamics

The frozen whitened embeddings define a trajectory. A train-only ridge model
predicts the current embedding from the previous three embeddings:

```text
e_hat(t) = B [e(t-1) || e(t-2) || e(t-3)] + b
r_dynamic(t) = mean((e(t) - e_hat(t))^2)
```

This measures whether the graph-state transition is unexpected, rather than
whether the raw one-step change is merely large.

Two label-free LogDet diagnostics are retained:

- prompt-channel volume: diversity/collapse of RP routing across layer/head
  channels;
- temporal spectral volume: local expansion/collapse of the recent embedding
  trajectory.

kNN distance in `[embedding || standardized innovation]` space is also kept as a
diagnostic, but it is no longer part of the primary score.

## 4. Train-only empirical-tail score

Static residual, dynamic residual, temporal volume deviation, and prompt volume
deviation are each converted to a position-conditioned empirical upper-tail
probability using only trimmed train references. The primary score is the mean
negative log tail probability over components available at that token:

```text
score_k(t) = -log p_train_tail,k(t)
score(t)   = mean_k score_k(t)
```

This avoids learning weights from labels and prevents an arbitrary z-score RMS
from letting a weak component dominate another component's scale.

## 5. Mechanism attribution

`test_scores.npz` stores the PCA residual split into RR and RP energy, plus the
top layer/head channels responsible for each token's off-subspace residual:

```text
rr_residual_energy
rp_residual_energy
top_channel_index
top_channel_energy
```

For a 32-head model, channel index `c` maps to
`layer = c // num_heads`, `head = c % num_heads`.

## Run

Default cache root:

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

Smoke test:

```bash
LIMIT=5 CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

Full fixed run:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

The runner performs exactly three stages:

```text
train attention -> reference.npz     labels never opened
 test attention -> test_scores.npz   labels never opened
 frozen scores  -> evaluation.json   labels opened here only
```

Default output:

```text
experiments/spectral_feasibility/outputs/spectral_subspace_dynamics/
```

`evaluation.json` reports the primary score and diagnostic components
independently. Component metrics are post-hoc analysis only and must not be used
to retune the already evaluated test split.
