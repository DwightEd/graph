# Evidence-Grounded Causal Attention Signal Audit

## Status

This directory is a **mechanism audit**, not a new final detector. It uses
retained prompt-to-response (PR) and response-to-response (RR) attention and
answers three questions before a more complex model is built:

1. Does the already reproduced PR/RR role shift survive without layer/head
   averaging?
2. What actually produced the historical RR residual signal?
3. Does hallucination onset exhibit premature concentration into a small,
   local response-history route set?

The audit does not assume that correct tokens possess a stable cross-layer/head
coordination structure. That hypothesis must pass explicit independent-vs-joint
and channel-shuffle controls before it can be stated.

No hallucination labels are used while extracting features, fitting references,
calibrating tails, or freezing test scores. Labels are opened only by the final
`evaluate` command.

## 1. Data contract

All attention enters through:

```python
open_research_dataset(...)
sample.iter_sparse_attention_blocks(...)
```

A retained RR edge is:

```text
response source j -> response target t
```

with exact `(layer, head, weight)` and `j < t`. Missing CSR entries mean only
`attention <= attention_floor`; the audit never interprets them as measured
zero-weight edges.

No missing edge is fabricated. PR and RR retained mass are divided by the
number of causally available sources before they are compared.

## 2. Evidence registry and channel-preserving role fields

The previous exact scalar audit is reproduced unchanged for five transparent
baselines. These are historical exploratory results, not independently held-out
feature selection:

| scalar | fixed anomaly direction | historical raw AUROC |
|---|---:|---:|
| history edge fraction | higher | 0.6418 |
| normalized entropy | lower | 0.4086 |
| history mass fraction | higher | 0.5865 |
| mean edge strength | higher | 0.5337 |
| direct Lookback anomaly | lower | 0.3899 |

The actual representation does not concatenate those five averages. For every
token and each channel `c=(layer,head)` it emits four separate `L×H` fields:

\[
P_{t,c}=\frac{\sum_{j\in prompt} A^c_{t,j}}{|prompt|},\qquad
R_{t,c}=\frac{\sum_{j<t} A^c_{t,j}}{\max(t,1)},
\]

\[
F_{t,c}=\frac{|E^{RR}_{t,c}|}{|E^{PR}_{t,c}|+|E^{RR}_{t,c}|},\qquad
S_{t,c}=\frac{\sum_{j\ne t} A^c_{t,j}}{|E^{PR}_{t,c}|+|E^{RR}_{t,c}|}.
\]

They are saved and modeled as separate 1024-dimensional blocks for a 32-layer,
32-head model. This prevents a weak feature from silently diluting a strong one
and makes every gain attributable to one mechanism family.

## 3. Decomposition of the historical mixed coordinate

For one channel `c=(layer,head)`, prefix `t`, and response source `j <= t`, let

```text
future[c,t,j] = sum_{u=j+1..t} A_c[u,j]
age[t,j]      = t-j+1
diag[c,j]     = A_c[j,j]
```

The historical implementation used:

\[
Q_{c,t,j}
=
\frac{\mathrm{future}_{c,t,j}}{\mathrm{age}_{t,j}}
-
\frac{\mathrm{age}_{t,j}-1}{\mathrm{age}_{t,j}}
\mathrm{diag}_{c,j}.
\]

It was previously described too loosely as a Laplacian eigenspectrum. It is
instead an artificial age-normalized source-persistence coordinate. This audit
separates it into four blocks:

### `received_topk`

\[
R_{c,t,j}
=
\frac{\mathrm{future}_{c,t,j}}{\mathrm{age}_{t,j}}.
\]

This tests whether the useful signal is the subsequent use of historical
response sources.

### `diagonal_topk`

\[
D_{c,t,j}
=
\frac{\mathrm{age}_{t,j}-1}{\mathrm{age}_{t,j}}
\mathrm{diag}_{c,j}.
\]

This tests whether the historical result was mostly a diagonal/self-attention
effect.

### `ratio_topk`

\[
P_{c,t,j}
=
\log\left(
1+
\frac{
\mathrm{future}_{c,t,j}/\max(\mathrm{age}_{t,j}-1,1)
}{
\mathrm{diag}_{c,j}+\epsilon
}
\right).
\]

This tests subsequent support relative to the source's original self-attention.

### `mixed_topk`

\[
Q=R-D.
\]

This exactly preserves the conceptual decomposition of the historical mixed
coordinate. Every block retains the strongest `K` values independently for all
layer/head channels. Only the mixed block uses strongest absolute magnitude and
keeps the original sign.

## 4. Current-token routing collapse

The historical prefix coordinates describe how old response sources continue
to be used. A separate block measures the current token's incoming RR routing
directly.

Per layer/head, the audit records:

```text
log retained RR mass
normalized retained-weight entropy
top-1 RR weight share
log effective RR source count
log mean causal lag
active-channel indicator
```

It also records interpretable all-channel variables:

```text
source entropy
effective exact-source count
source top-1 share
mean lag
local-lag mass share
top-source anchor turnover
lag-route velocity
active-channel fraction
route effective rank
cross-channel lag-profile consensus
```

The preregistered premature-collapse hypothesis predicts:

```text
source entropy                 lower
effective source count         lower
source top-1 share             higher
mean lag                       lower
local-lag mass share           higher
anchor turnover                lower
lag-route velocity             lower
route effective rank           lower
```

`cross_channel_consensus`, total RR mass, and edge count remain diagnostics;
their directions are not treated as established.

## 5. What “coordination” must prove

A global PCA residual being better than one peak head does not prove
cross-channel coordination. The audit therefore fits two densities to every
block:

### Independent model

A conditionally standardized diagonal Gaussian:

\[
p_{\mathrm{ind}}(x)
=
\prod_k \mathcal N(x_k;\mu_k,\sigma_k^2).
\]

### Joint factor model

A robust PCA/PPCA factor model:

\[
p_{\mathrm{joint}}(x)
=
\mathcal N
\left(
x;
\mu,
U\Lambda U^\top+\sigma^2I
\right).
\]

Both are fitted on the same unlabeled fit rows. Their held-out test scores are
reported separately. If the joint model does not outperform the independent
model, there is no evidence that cross-channel dependence helps detection.

The unlabeled calibration stream also supplies a channel-shuffle null. Within
the same task and position condition, each channel block is independently
permuted across token rows. This preserves every conditional channel marginal
but destroys row-wise alignment across channels. The audit reports:

\[
\Delta_{\mathrm{coord}}
=
NLL_{\mathrm{joint}}(X_{\mathrm{shuffle}})
-
NLL_{\mathrm{joint}}(X_{\mathrm{real}}).
\]

A positive source-group bootstrap interval establishes only that real channel
alignment is statistically learnable. It does **not** establish that the
alignment is related to correctness; label-based metrics are reported
separately.

## 6. Conditioning and temporal scope

Two condition schemes are frozen:

### `relative`

```text
task_type × final-response-relative position bin
```

This reproduces the historical offline protocol but uses final response length.

### `causal`

```text
task_type × floor(log2(token_index+1))
```

This uses only the current causal prefix. Comparing the two exposes whether the
historical result depends on final-length conditioning.

## 7. Scores and outputs

For every signal block and both condition modes, the audit freezes:

```text
residual_tail
independent_nll_tail
ppca_nll_tail
```

For collapse variables, it freezes predeclared one-sided tails, two-sided tails,
and a fixed equal-weight collapse composite.

Evaluation writes:

```text
evaluation/evaluation.json
evaluation/score_metrics.csv
evaluation/onset_effects.csv
```

`score_metrics.csv` is a feature-discovery table. It reports fixed-direction
AUROC/AUPRC plus orientation-free AUROC as an explicitly post-hoc diagnostic.
No best component is silently promoted to a final detector.

## 8. Decision rules

The next method is allowed to claim a cross-channel mechanism only if all of the
following hold on held-out source groups:

1. joint PPCA is better than independent marginals;
2. real channel alignment is better than the conditional channel-shuffle null;
3. the coordination-only score has stable label separation;
4. the result persists in same-sample hallucination-onset analysis.

The premature-collapse direction is supported only if the raw onset effects
jointly show lower entropy/effective source count/lag/turnover/velocity and
higher top-1/local mass.

Otherwise the corresponding narrative must be removed.
