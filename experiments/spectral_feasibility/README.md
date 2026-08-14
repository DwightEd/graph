# Causal dual-spectrum experiment

This experiment replaces the discarded `channel-mean -> AA^T -> HKS/SVD -> 22D`
feasibility baseline. The old construction averaged layer/head channels before
spectral analysis, symmetrized the directed attention flow, and normalized away
important attention scale information. Its near-random full-test result is kept
only as a negative design lesson; the implementation itself is no longer the
active spectral method.

## Literature boundary

Two different spectral ideas motivate this experiment and must not be conflated.

- **LapEigvals** (*Hallucination Detection in LLMs Using Spectral Features of
  Attention Maps*, EMNLP 2025) directly treats every layer/head attention map as
  a directed weighted adjacency matrix. It constructs `L = D - A`; because
  causal attention is triangular, the Laplacian eigenvalues are its diagonal.
  The official implementation keeps the largest `k` eigenvalues independently
  for every layer/head and only then concatenates channels. Its published probe
  is supervised and response-level.
- **EigenScore** (*INSIDE: LLMs' Internal States Retain the Power of Hallucination
  Detection*, ICLR 2024) does **not** use an attention-graph Laplacian. It takes
  a covariance matrix of sentence embeddings from multiple generated responses
  and uses a regularized LogDet / sum of log eigenvalues to measure occupied
  semantic volume. We borrow only this LogDet-volume principle for label-free
  spectral-state diagnostics.

Our task is different from both: token-level, fully unsupervised, and based on a
sparse response-query cache. The method below adapts their useful principles
without pretending that unavailable attention entries exist.

## Data boundary

Every raw-attention read goes through `research_dataset.open_research_dataset()`
and `ResearchSample`. Experiment code must not open canonical NPZ or formal PT
files directly.

The formal cache provides:

- exact stored attention diagonals for all layer/head channels;
- retained response-query edges to earlier prompt/response sources;
- no prompt-query rows;
- no exact values for off-diagonal entries at or below `attention_floor`.

Therefore all spectra below are explicitly spectra of the **cache-censored
operator**. Missing values are not claimed to be original zeros.

## 1. RR causal directed Laplacian spectrum

For every channel `c=(layer, head)`, response prefix ending at response-relative
token `t`, and response source node `j <= t`, define

```text
d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
lambda[c,t,j] = d[c,t,j] - A_c[j,j]
```

This is the no-vertical-edge LapEigvals degree convention restricted to the
response subgraph. The restriction is deliberate: all query rows that can point
to a response node are response-query rows and are available in the cache. A
full prompt+response Laplacian cannot be reconstructed because prompt-query rows
were never stored.

The causal response Laplacian is lower triangular, so its eigenvalues are the
`lambda[c,t,j]` values. At every token we keep the largest `k` values for **each
channel separately**:

```text
s_RR(t) = concat_{layer,head} TopK(lambda[c,t,:])
```

No layer/head mean is taken before this operation. With 32 layers, 32 heads and
`k=5`, this block alone is 5120-dimensional.

## 2. RP prompt transport spectral state

Ignoring prompt routing would throw away a known important part of the attention
operator. But a prompt-node Laplacian would require prompt-query rows that do not
exist. We therefore use the mathematically supported rectangular transport
operator instead of fabricating edges.

For every response token and channel, prompt-source attention is projected by a
fixed CountSketch. Coordinate zero is not hashed: it is the exact **retained
prompt mass**. The remaining coordinates are deterministic signed hashes of
prompt source positions:

```text
y[c,t,0] = sum_{j in prompt} A_c[t,j]
y[c,t,b(j)] += sign(j) A_c[t,j]
```

This preserves scale plus a low-dimensional approximation of which prompt
sources are used while retaining every layer/head as a separate block. With the
default sketch dimension `m=4`, this contributes another 4096 dimensions.

The raw token state is therefore

```text
s(t) = [ s_RR(t) || y_RP(t) ]
```

and is 9216-dimensional for a 32x32 attention geometry with default settings.

## 3. EigenScore-inspired channel spectral volume

At token `t`, reshape prompt sketches to `Y_t in R^{C x m}`, where `C=L*H`.
Across channels we compute

```text
Sigma_t = centered(Y_t)^T centered(Y_t) / C + alpha I
V_prompt(t) = (1/m) log det(Sigma_t)
```

This asks whether the prompt-routing patterns occupied by the attention channels
collapse or expand. It is **inspired by** EigenScore's covariance LogDet, not
claimed to be EigenScore itself.

## 4. Label-free normal spectral manifold

Training labels are never opened. Four approximately uniform causal prefixes
per train response are used as reference states by default.

Raw 9216-D spectra are robustly normalized within response-position bins using
train median/MAD. Unsupervised randomized PCA then learns cross-layer/head
combinations and whitens them:

```text
e(t) = whiten(PCA( robust_standardize(s(t)) ))
```

Default dimension is 32. PCA is not trained with hallucination labels.

The one-step spectral innovation is

```text
Delta e(t) = e(t) - e(t-1)
```

and the main normal-manifold vector is

```text
q(t) = [ e(t) || robust_standardize(Delta e(t)) ]
```

which is 64-D with default settings.

A task- and relative-position-conditioned k-nearest-neighbor reference is fitted
only on train `q(t)` vectors. This avoids forcing normal tokens into one Gaussian
ellipsoid and does not require an assumption that anomalies form a minority
cluster of a particular shape.

## 5. Temporal spectral volume

A second LogDet diagnostic uses the recent learned spectral trajectory. For the
last `w` embeddings, form `Z` and

```text
G_t = centered(Z)^T centered(Z) / d + alpha I
V_time(t) = (1/w) log det(G_t)
```

It measures local expansion/collapse of graph-spectrum dynamics. Again this is
an EigenScore-inspired volume principle applied to a different object.

## 6. Frozen unsupervised score

Train data calibrates four quantities by position-bin robust statistics:

1. distance to the conditional normal manifold (`kNN`);
2. PCA off-subspace reconstruction residual;
3. temporal spectral-volume deviation;
4. prompt channel-volume deviation.

Their calibrated deviations are combined with a fixed equal-weight RMS. No
weight, sign, threshold, PCA dimension, neighbor count, or component is selected
from test hallucination labels.

The score artifact additionally stores `embedding` and standardized
`innovation`. These are the node-level vectors to visualize when asking whether
hallucination tokens share a repeatable spectral anomaly direction.

## Run

The default data root is the existing RAGTruth-derived formal attention cache:

```text
/share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/
outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876
```

First run a smoke test:

```bash
LIMIT=5 DEVICE=cuda bash experiments/spectral_feasibility/run.sh
```

Then run the fixed configuration on the complete split:

```bash
DEVICE=cuda bash experiments/spectral_feasibility/run.sh
```

The runner performs only three stages:

```text
train attention -> fit reference.npz          labels never opened
 test attention -> test_scores.npz            labels never opened
 frozen scores  -> evaluation.json            labels opened here only
```

Default output:

```text
experiments/spectral_feasibility/outputs/causal_dual_spectrum_v1/
  reference.npz
  test_scores.npz
  evaluation.json
```

`evaluation.json` reports the fixed combined score and each component separately
(`manifold_knn`, `pca_residual`, `temporal_spectral_volume`,
`prompt_channel_volume`, `innovation_norm`). Component reporting is diagnostic:
it tells us which spectral hypothesis survived, but it must not be used to tune
the already-run test experiment.
