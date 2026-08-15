# RR causal spectral-subspace anomaly detection

This experiment is the active fully unsupervised spectral detector.  Its primary
hypothesis is deliberately narrow:

> Hallucination tokens can leave the low-dimensional normal subspace of the
> **response-history (RR) causal attention spectrum**.

The primary detector therefore uses RR only.  RP routing, temporal dynamics and
kNN no longer enter the score after full-data diagnostics showed that mixing
weak components can dilute the RR signal.  Labels are never used by fitting or
scoring; they are opened only by the final `evaluate` command.

## Data boundary

All attention access goes through `research_dataset.open_research_dataset()` and
`ResearchSample.iter_sparse_attention_blocks()`.  Experiment code never parses
canonical NPZ/PT files directly.  Missing sparse-cache entries remain censored
(`<= attention_floor`); they are not reconstructed as exact zeros.

## 1. RR token-aligned causal spectrum

For each response token `t` and channel `c=(layer, head)`, retained RR edges form
a causal response-prefix graph.  Following the existing LapEigvals degree
convention,

```text
d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
lambda[c,t,j] = d[c,t,j] - A_c[j,j]
```

The causal Laplacian is triangular, therefore its diagonal is its spectrum.  We
keep the `top_k` eigenvalues with largest absolute magnitude independently for
every layer/head while preserving sign:

```text
x_RR(t) = concat_{layer,head} StrongestAbsK(lambda[c,t,:])
```

For a 32-layer x 32-head model with `top_k=5`:

```text
D_RR = 32 * 32 * 5 = 5120
```

No layer/head averaging occurs before the subspace model.

## 2. RR-only robust normal subspace

Six approximately uniform reference tokens are taken from every train response
by default.  Their 5120-D RR states are normalized within four relative response
position bins using train-only median/MAD:

```text
z_t = (x_RR(t) - median_bin) / MAD_bin
```

A provisional PCA is fitted only to unlabeled train references.  Within each
position bin the largest 10% provisional reconstruction-residual tail is removed,
and PCA is refitted on the remaining references:

```text
P_RR = PCA(trimmed train RR states)
e_t  = whiten(P_RR z_t)
eps_t = z_t - reconstruct_RR(z_t)
r_global(t) = mean(eps_t ** 2)
```

The **primary anomaly score** is only the position-conditioned empirical upper
tail of `r_global` measured on the retained train references:

```text
score(t) = -log p_train( r_global >= r_global(t) | position_bin )
```

No other diagnostic is fused into `score`.

## 3. Fixed ablations and channel localization

The same run freezes two comparisons before labels are opened:

1. `rr_untrimmed_pca_ablation`: identical RR input but PCA fitted without the
   10% trimming step.  This tests whether robust trimming itself helps.
2. `rr_localized_channel_tail`: reshape the residual as
   `[layer*head, top_k]`, compute one residual energy per channel, standardize
   each channel against its own train distribution, and average only the fixed
   top 5% channel tail.  This tests the hypothesis that an abnormal token may be
   concentrated in a small subset of heads rather than diffuse over all 1024
   channels.

For channel `c`:

```text
r_c(t) = mean_k eps[t,c,k]^2
z_c(t) = max((r_c(t) - median_train[c,bin]) / MAD_train[c,bin], 0)
r_local(t) = mean(Top5Percent_c z_c(t))
```

`r_local` receives its own train-only empirical-tail calibration.  It is stored
as a diagnostic and **does not enter the primary score**.

`test_scores.npz` also stores the most abnormal channel indices and calibrated
channel scores.  For `num_heads=32`, channel `c` maps to:

```text
layer = c // 32
head  = c % 32
```

This allows later onset/layer/head mechanism analysis without recomputing the
attention spectra.

## 4. Label firewall

The execution contract is:

```text
train attention -> RR reference.npz    labels never opened
 test attention -> test_scores.npz     labels never opened
 frozen scores  -> evaluation.json     labels opened here only
```

The fit artifact and score artifact contain no hallucination labels.  Evaluation
may need to scan the complete test cache before the formal label store unlocks;
that scan now has an explicit progress bar.

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

Smoke output is isolated automatically:

```text
experiments/spectral_feasibility/outputs/rr_spectral_subspace/smoke_5/
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

Full output:

```text
experiments/spectral_feasibility/outputs/rr_spectral_subspace/full/
```

Set `OUT=/custom/path` to override either location.

The final `evaluation.json` reports the primary RR-only score together with the
untrimmed-PCA and localized-channel fixed ablations.  These post-hoc metrics are
for diagnosis; changing the detector after reading them requires a fresh held-out
split for a clean final claim.
