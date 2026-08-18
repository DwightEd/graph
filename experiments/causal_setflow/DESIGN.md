# Mechanism-Guided Causal Attention Set-Flow

## Status

The scalar-imputation CASF prototype is retired as the active method. Its
faithful smoke run produced token AUROC `0.5012` and AUPRC `0.03455`, while the
strict causal RR received-support baseline reached AUROC `0.6784` and AUPRC
`0.1375` on the full audit. The redesign therefore changes the learning target
and anomaly score rather than merely increasing epochs or model size.

The active method is **Mechanism-Guided Causal Attention Set-Flow (MG-CASF)**.
It remains label-free with respect to hallucination labels.

## 1. Scientific object

For response token `t`, Transformer layer `l`, and head `h`, retained RR
attention defines a weighted source set

\[
E_{t,l,h}=\{(j,A^{l,h}_{t,j}):j<t\}.
\]

The model preserves four nested structures:

1. source members inside one weighted attention set;
2. heads inside one Transformer layer;
3. ordered Transformer depth;
4. autoregressive token time.

The exact response source index is used only inside a response to gather causal
ancestry state. It is never treated as a cross-sample semantic ID.

## 2. Why the previous objective failed

The retired objective revealed the source member and asked the network to
interpolate three scalar fields: edge weight, received support, and received
support delta. It did not require the encoder to learn set membership, complete
source-set shape, or a hallucination-relevant anomaly direction. Layer/head
errors were averaged, and six unrelated upper-tail scores were combined with an
equal-weight Fisher statistic. This discarded the validated channel field and
could not represent anomalies that are unusually easy or unusually stable.

## 3. Encoder

The encoder keeps the memory-efficient implementation and returns both token
and channel fields:

```text
source fields
  -> Set Transformer source-set encoder
  -> head interaction
  -> exact-source depth recurrence (SetWalk ancestry)
  -> ordered depth mixer
  -> causal token-time encoder
```

For every token it exposes

\[
C_t\in\mathbb R^{L\times H\times D},\qquad z_t\in\mathbb R^D,
\]

where `C_t` is never reduced to a scalar before anomaly learning.

Typed source fields are encoded by separate branches: causal lag, current edge
weight, age-normalized received support, received-support delta, and ancestry
state. They are fused after type-specific encoding; raw statistics are not
flattened and concatenated into a single hand-designed feature vector.

## 4. EMA clean teacher

A student encoder is paired with a stop-gradient exponential-moving-average
teacher. The teacher sees the complete clean source sets. The student sees a
label-free stochastic view with source members or channel groups hidden.

Student predictions are aligned to teacher token and channel states:

\[
\mathcal L_{token}=1-\cos(p_z(z_t^{student}),\operatorname{sg}(z_t^{teacher})),
\]

\[
\mathcal L_{channel}=\operatorname{mean}_{l,h}
\left[1-\cos(p_c(C_{t,l,h}^{student}),
\operatorname{sg}(C_{t,l,h}^{teacher}))\right].
\]

The EMA target prevents a moving reconstruction target. Variance and covariance
regularization are applied to token and channel representations to prevent the
near-constant latent state observed in the retired prototype.

## 5. Received-support preservation

The full audit showed that subsequent received support, rather than the old
mixed diagonal subtraction, is the strongest stable RR-only signal. MG-CASF
therefore includes a structured auxiliary decoder from each channel state to
that channel's ordered received-support memory signature. This is an auxiliary
representation constraint, not the final detector.

The objective preserves the full layer/head field:

\[
\widehat R_t\in\mathbb R^{L\times H\times K},
\]

and does not average reconstruction residuals over channels during training or
scoring.

## 6. Mechanism-guided synthetic anomalies

Hallucination labels are never used. Instead, the model learns an anomaly
orientation from causal, domain-specific graph corruptions derived from the RR
audit. Each corruption returns the exact affected token mask.

### Local route collapse

Concentrate a row's retained mass onto one or a few recent response sources,
while preserving target token, layer, head, total row mass, and causal validity.

### Lag-preserving source rewire

Change exact response source identities inside the same coarse causal-lag
regime while preserving edge weights and row marginals. This isolates ancestry
and exact-source structure.

### Temporal route freeze

Copy the previous token's source-set pattern into the current token, preserving
causality. This creates excessive recurrence and low route velocity.

### Head homogenization

Make several heads share one donor source-set pattern, with recipient row mass
rescaling. This simulates loss of head specialization without simply changing
total attention mass.

### Received-memory permutation

Keep the multiset of received-support values but permute their exact source
assignments, breaking source-persistence ancestry while preserving the marginal
value distribution.

These corruptions are not claimed to be hallucinations. They provide
mechanism-specific pseudo anomalies whose transfer to real hallucination must be
verified by held-out evaluation and ablation.

## 7. Mechanism-query energy head

The anomaly head does not concatenate six scalar detector scores. It uses one
learned query per corruption mechanism to attend over the complete channel field
`C_t`. Each expert produces one corruption logit. Token state and causal temporal
change modulate expert context through learned projections.

For mechanism `m`:

\[
e_{t,m}=g_m\left(
\operatorname{Attn}(q_m,\{C_{t,l,h}\}_{l,h}),
 z_t, z_t-z_{t-1}
\right).
\]

Clean tokens are trained as negative examples. Tokens affected by corruption
`m` are positive only for expert `m`. A ranking margin additionally requires
corrupted energy to exceed the corresponding clean energy.

The primary raw score is fixed before labels open:

\[
E_t=\log\sum_m\exp(e_{t,m}).
\]

An independent unlabeled calibration split converts this oriented energy to an
empirical upper-tail score conditioned on task and causal token position. No
post-hoc sign inversion, component selection, or Fisher fusion is used.

## 8. Recovery and drift diagnostics

The score artifact also keeps, without promoting them to the primary detector:

- teacher-student token recovery discrepancy;
- teacher-student channel recovery discrepancy;
- received-signature reconstruction error by layer/head;
- mechanism expert logits;
- latent temporal speed and curvature;
- clean-versus-corrupted ranking margins.

This follows the useful distinction in temporal anomaly detection between
state recovery and state drift, while retaining attention-specific source-set
geometry.

## 9. Training protocol

```text
unlabeled fit source groups
  -> optimize student, EMA teacher, corruption experts

unlabeled calibration source groups
  -> freeze empirical energy distributions

held-out test source groups
  -> freeze token energy and component artifacts without labels

post-hoc evaluate
  -> open token labels and compute AUROC/AUPRC
```

Complete `source_id` groups remain disjoint. The final score is causal and does
not use final response length.

## 10. Required ablations

The method is supported only if it passes all of the following on identical
held-out token rows:

1. MG-CASF primary energy exceeds the strict causal received-support baseline;
2. EMA teacher is better than a moving self-target;
3. mechanism-guided corruptions are better than generic random feature noise;
4. channel-field energy is better than layer/head averaging;
5. exact-source ancestry is better than no SetWalk ancestry;
6. Set Transformer is better than DeepSets pooling;
7. temporal route-freeze and local-collapse experts add complementary signal;
8. source-group bootstrap confidence intervals for improvement exclude zero.

The target reference is at least AUROC `0.6784` and AUPRC `0.1375` on the
current strict causal protocol. A smoke subset is used only for runtime and
mechanism sanity checks.

## 11. Module responsibilities

```text
config.py       scientific and execution configuration
corruptions.py  causal mechanism corruption plans and invariants
data.py         canonical sparse RR -> bounded weighted source sets
set_layers.py   permutation-invariant set-attention primitives
model.py        student/EMA encoder and structured outputs
energy.py       mechanism-query channel-field energy head
losses.py       EMA recovery, ranking, variance, covariance objectives
trainer.py      label-free optimization and EMA updates
calibration.py  oriented energy calibration only
artifacts.py    strict checkpoint/reference/score schemas
experiment.py   fit, score, and post-hoc evaluation protocol
main.py         CLI
run.sh          one-command smoke/full workflow
```
