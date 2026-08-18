# Mechanism-Guided Causal Attention Set-Flow

## Status

This directory implements a label-free token-level hallucination detector over
retained response-to-response (RR) attention.  The method is no longer the V1
scalar-imputation detector.  V1 demonstrated that a faithful Set-Flow encoder
can be trained within memory limits, but its equal-weight upper-tail score was
random on the smoke split and its latent representation collapsed.

The active method is **Mechanism-Guided Causal Attention Set-Flow (MG-CASF)**.
It learns an anomaly-energy direction from causal structural corruptions rather
than assuming that every reconstruction error must be large for hallucination.
No hallucination label is opened during representation learning, corruption
training, calibration, or held-out scoring.

## 1. Empirical starting point

The redesign is anchored to two full-split RR observations:

1. age-normalized received support across layer/head channels is a stronger
   signal than the old mixed received-minus-diagonal coordinate;
2. hallucination onset is accompanied by lower source entropy, fewer effective
   sources, higher top-1/local mass, lower anchor turnover, lower route velocity,
   and lower route effective rank.

The second observation means that hallucination can be **too regular and too
stable**.  It is therefore incorrect to equate anomaly with high prediction or
reconstruction error only.

## 2. Data object: a measure-valued causal source set

For response token `t`, Transformer layer `l`, and attention head `h`, the
retained RR row is the weighted set

\[
E_{t,l,h}=\{(j,A^{l,h}_{t,j}):j<t\}.
\]

The model also keeps a received-support memory set whose member value is

\[
R^{l,h}_{t,j}=\frac{\sum_{u=j+1}^{t}A^{l,h}_{u,j}}{t-j+1}.
\]

Each source member is encoded from typed fields rather than a flat feature
vector:

- causal lag;
- current attention weight;
- received support;
- received-support change;
- the exact source token's previous-layer Set-Flow state.

A Set Transformer encodes members within one weighted source set.  A head mixer
models same-layer head interactions, a causal depth recurrence transports exact
source ancestry to the next layer, an ordered depth mixer summarizes the layer
trajectory, and a causal time encoder produces token states.

## 3. Why V1 was replaced

V1 masked scalar fields of an already-known source member and reconstructed only
`weight`, `received`, and `received_delta`.  This did not require learning set
membership, ancestry degeneracy, head specialization, or dynamic collapse.  It
also averaged layer/head error fields and combined six upper-tail scores with an
equal Fisher sum.  These choices discarded the strongest known channel-preserving
signal and imposed the wrong anomaly direction.

MG-CASF removes scalar imputation as the primary task.  The encoder is trained
against structured source-set corruptions that preserve nuisance quantities but
alter the mechanisms implicated by the RR audit.

## 4. Mechanism-guided causal corruptions

One training example selects a contiguous response-token span, a contiguous
Transformer-layer band, and a non-empty subset of heads.  One of five corruption
families is applied only inside that causal block.

### 4.1 Concentration collapse

For valid weights `w_i`,

\[
w'_i=\frac{w_i^\gamma}{\sum_k w_k^\gamma}\sum_k w_k,\qquad \gamma>1.
\]

The source set, support count, and total mass are preserved while entropy falls
and top-1 share rises.

### 4.2 Local-source contraction

The weight multiset and member count are preserved, but exact source identities
are replaced by the most recent legal causal sources.  This changes lag and
ancestry without creating future edges.

### 4.3 Route freezing

A selected token inherits the previous token's source-set pattern, rescaled to
its own retained mass.  This creates low turnover and low route velocity without
changing the token count, layer count, or head count.

### 4.4 Cross-head homogenization

Selected heads inherit one anchor head's source identities and normalized
pattern while retaining their own row mass.  This simulates loss of head
specialization and anchor coalescence.

### 4.5 Received-support self-reinforcement

Current route weights are reweighted toward members with high received support,
while total route mass is preserved.  This creates a self-reinforcing
response-history attractor.

The corruption engine never uses token labels.  Corruption type is a synthetic
training target only.

## 5. Online encoder and EMA teacher

Let `f_theta` be the online Set-Flow encoder and `f_xi` an exponential-moving-
average teacher:

\[
\xi\leftarrow m\xi+(1-m)\theta.
\]

For a clean graph `G` and its corrupted view `G^-`, the model computes

\[
z=f_\theta(G),\quad z^-=f_\theta(G^-),\quad \bar z=f_\xi(G).
\]

The online clean state predicts the stop-gradient teacher state.  Corrupted
states are required to recover teacher context outside the corrupted block.
Variance and covariance regularizers prevent representation collapse.  Unlike
V1, the anti-collapse objective is a first-class loss rather than a tiny
auxiliary penalty.

## 6. Learned anomaly energy

MG-CASF predicts anomaly energy at two resolutions:

- one energy per `(token, layer, head)` channel state;
- one energy from token state, velocity, and curvature.

Channel energies are aggregated with log-mean-exp, not averaged, so distributed
or localized channel responses are retained.  A learned gate combines channel
and token energies.  Five specialized heads predict the synthetic corruption
family, while one general head is trained against all corruption families.

For paired clean and corrupted tokens, the general energy obeys

\[
E(G^-_t)\ge E(G_t)+\delta.
\]

Clean energy uses a trimmed loss: the highest-loss fraction is excluded from the
clean target so that an unlabeled training split containing a small number of
real anomalies is not forced entirely into the low-energy class.

The primary held-out score is the conditionally calibrated upper tail of the
**single learned general energy**.  There is no post-hoc sign choice and no
equal-weight Fisher fusion.  Type-specific energies are frozen diagnostics.

## 7. Loss

The total label-free objective is

\[
\mathcal L=
\lambda_c\mathcal L_{clean-energy}
+\lambda_a\mathcal L_{corrupt-energy}
+\lambda_r\mathcal L_{ranking}
+\lambda_t\mathcal L_{type}
+\lambda_{ema}\mathcal L_{clean-recovery}
+\lambda_{ctx}\mathcal L_{context-recovery}
+\lambda_v\mathcal L_{variance}
+\lambda_{cov}\mathcal L_{covariance}.
\]

All targets are generated from attention itself or from the EMA teacher.  No
hallucination label contributes to the gradient.

## 8. Calibration and evaluation

Complete source groups are split into label-blind fit and calibration streams.
The general energy and each diagnostic energy are conditionally calibrated by

```text
task_type × floor(log2(token_index + 1))
```

using finite-sample empirical upper tails.  Test rows are frozen before labels
are opened.  The score is online-causal and does not use final response length.

## 9. Required ablations

The method is supported only if the following comparisons hold on identical
held-out rows:

- full MG-CASF vs received-support causal residual baseline;
- full model vs no exact-source ancestry;
- Set Transformer vs DeepSets pooling;
- full corruption bank vs each corruption family removed;
- channel energy vs token-only energy;
- EMA teacher vs no teacher;
- learned general energy vs V1 scalar-imputation score;
- real layer order vs layer-shuffled control.

A high synthetic-corruption AUC alone is not evidence of hallucination
detection.  Improvements must be reported with source-cluster bootstrap
intervals and task-specific results.

## 10. Claim boundary

MG-CASF is label-free in optimization and calibration, but its corruption bank
was designed after an exploratory analysis of hallucination mechanisms.  A
final scientific test therefore requires source groups or datasets not used in
that exploratory model design.