# RR causal spectral geometry

This directory contains one attention-only, label-free representation and
anomaly detector. Its narrow hypothesis is that a generated token can be
abnormal when the causal response-history (RR) routing prefix leaves the
dominant geometry of unlabeled train prefixes.

It does not train a GNN, average layer/head channels, fabricate missing edges,
or mix prompt and response statistics into one score. Labels are opened only by
the final evaluation command.

## Representation

For layer/head channel `c`, current response prefix `t`, and response source
`j <= t`, define the artificial age-normalized coordinate

```text
d_age[c,t,j] = sum_{u=j..t} A[c,u,j] / (t-j+1) - A[c,j,j]
```

`A[c,u,j]` is the retained response-to-response attention from query `u` to
source `j`. This defines an artificial age-normalized triangular attention
operator, not a standard graph Laplacian. The code uses its diagonal
coordinates directly; there is no eigendecomposition or eigenvector rotation.
It keeps the signed `top_k` values with largest magnitude in every layer/head
independently:

```text
x(t) = concat_{layer,head} StrongestAbsK(d_age[c,t,:])
```

For 32 layers, 32 heads, and `top_k=5`, `x(t)` has 5120 coordinates. This is a
dynamic prefix descriptor assigned to the newly generated token `t`; it does
not claim to model hidden-state message passing.

All attention is read through `research_dataset.py`. Missing cache entries stay
censored below `attention_floor`; PP edges are neither needed nor invented.

## Fit and calibration protocol

Complete source groups are deterministically split into two disjoint unlabeled
streams:

```text
fit groups          -> position median/MAD -> two-pass robust PCA
calibration groups  -> frozen transform    -> empirical score distributions
test groups         -> source-overlap audit -> frozen transform -> scores
```

No token from a calibration source is used to fit the scaler or PCA. Position
is controlled once, using fit-only median/MAD in four relative-position bins.
A provisional PCA then removes the fixed upper 10% fit-residual tail as an
unlabeled contamination guard, and one final PCA is fitted. Channel scales are
also fitted on this retained fit stream. Calibration rows are never filtered
or used to fit a transform; they only define empirical score distributions.

The canonical score is the global finite-sample upper-tail probability of
orthogonal reconstruction energy:

```text
r_perp(t) = mean((z(t) - PCA^-1(PCA(z(t))))^2)
score(t)  = -log p_calibration(r_perp >= r_perp(t))
```

The global calibration is intentionally monotone in `r_perp`. There is no
second position-conditioned remapping after position standardization.

## Mechanism diagnostics

The score artifact preserves four complementary readings of the same geometry:

- `rr_residual_energy`: orthogonal escape from the dominant subspace;
- `rr_latent_energy`: extreme location inside that subspace;
- `rr_ppca_energy`: the PPCA quadratic form combining the two with fitted
  variances;
- `rr_localized_residual`: escape concentrated in a small layer/head subset.

Only the first is the primary detector. The other three are reported separately
and are never averaged or selected using test labels. The top residual channels
are stored for later layer/head/source/lag attribution.

## Files

- `representations.py`: causal RR operator and signed per-channel modes;
- `subspace.py`: robust PCA geometry and empirical calibration;
- `artifacts.py`: schemas, dimensional checks, and strict artifact loading;
- `experiment.py`: fit, score, label-firewalled evaluation, and artifacts;
- `main.py`: CLI;
- `run.sh`: one foreground runner.

## Run

First run a meaningful smoke test. Training and test limits are deliberately
separate; the former `LIMIT=5` interface is rejected because five training
samples can make PCA interpolate the smoke data.

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
TRAIN_LIMIT=200 TEST_LIMIT=5 CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

Then run the full experiment:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/spectral_feasibility/run.sh
```

Default full output:

```text
experiments/spectral_feasibility/outputs/rr_spectral_subspace_v2/full/
  reference.npz
  test_scores.npz
  evaluation.json
```

The v2 reference and score schemas are intentionally incompatible with earlier
artifacts. Score files persist the fit/calibration/test source audit and whether
the tested sample scope is complete or limited. They also bind the exact
on-disk test `manifest.json` digest and store `response_length` on every complete
`0..R-1` response row. Evaluation reloads the captured score path and verifies
its digest, manifest identity, test split, canonical source, response length,
and complete token coverage before it unlocks labels.
The versioned default output directory prevents new runs from mixing with old
artifacts; `OUT=/new/path` can still select another fresh directory.
