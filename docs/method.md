# Label-free causal topology representation

This experiment asks whether the *arrangement* of retained causal attention
contains token-level hallucination signal beyond simple attention marginals.
It has no learned message-passing network and reads no labels while encoding,
fitting references, or producing anomaly scores. Labels are opened only after
the score artifact is frozen, for AUROC/AUPRC and paired-bootstrap evaluation.

## 1. Sparse attention semantics

Let the cache threshold be \(f=0.01\). A missing CSR entry is not an edge with
weight \(0.01\). Instead, \(0.01\) is the implicit background used only to
reconstruct attention *marginals* over all eligible causal sources:

\[
M^P_{t,c}=|P|f+\sum_{s\in P\cap E_{t,c}}(a_{t,s,c}-f)_+,
\qquad
M^R_{t,c}=t f+\sum_{s<t,\,s\in E_{t,c}}(a_{t,s,c}-f)_+.
\]

Here \(c=(l,h)\) is one exact layer-head channel. The diagonal is read from its
stored dense field. The channel state keeps both radial quantities that the old
ratio discarded:

\[
x_{t,c}=\left[
\frac{M^P_{t,c}/|P|}{M^P_{t,c}/|P|+(M^R_{t,c}+d_{t,c})/(t+1)},
\log\!\left(M^P_{t,c}/|P|+(M^R_{t,c}+d_{t,c})/(t+1)\right)
\right].
\]

Topology uses only retained **excess** weights
\(e_{t,s,c}=(a_{t,s,c}-f)_+\). Therefore a threshold-valued entry contributes
to retained-support diagnostics but does not invent a routed message.

## 2. Prompt provenance without prompt-length pooling

A prompt source position \(s\) is encoded by \(K\) Fourier frequencies:

\[
\phi_K(s)=\left[\sin(2\pi ks/|P|),\cos(2\pi ks/|P|)\right]_{k=1}^{K}.
\]

For every token and layer-head channel, the encoder stores the excess-weighted
mean of \(\phi_K(s)\). This preserves coarse and fine prompt-source location
while keeping a fixed \(2K\)-coordinate interface for different prompt
lengths. `FOURIER_FREQUENCIES` controls \(K\); the default is 4.

## 3. Response-history topology

For each response target and channel, excess RR weights are normalized over its
retained history neighbors. The encoder does not mean-pool layers or heads. It
emits, per channel:

- the normalized neighbor mean of the two-coordinate state \(x\);
- the normalized mean absolute target-neighbor difference;
- the normalized neighborhood variance;
- a second-hop neighbor mean obtained by routing the first-hop means once more.

Signed one-hop differences can cancel even when a neighborhood is abnormal.
The absolute difference and variance retain that dispersion, while the second
hop distinguishes similar marginals produced by different causal paths.

## 4. Lag-matched topology null

The control graph keeps every RR target, channel, excess weight, and causal lag
scale, but rewires the source within the same
\(\lfloor\log_2(\mathrm{lag})\rfloor\) bin. Prompt routes and all attention
marginals remain unchanged. Comparing exact RR propagation with this lag
rewire tests source arrangement, rather than degree, mass, or short-versus-long
lookback alone.

## 5. Atomic one-class references

Each scalar topology coordinate remains a separate
`[token, layer * head]` block. The method never concatenates a few graph
coordinates beside a much wider vector and then lets PCA wash them out. Atomic
families are:

- balance and log total scale;
- prompt, response-history, and diagonal marginals;
- retained prompt and history support;
- each prompt Fourier coordinate;
- each exact and lag-rewired RR mean, absolute-difference, variance, and
  second-hop coordinate.

Unlabeled train source groups (or sample IDs when no source ID exists) are
assigned by a stable hash to disjoint `fit` and `cal` streams. `REFERENCE_SIZE`
is their **combined** token budget; each stream receives half. A position-bin
bottom-k reservoir uses the same priorities and slots for every atomic block,
so all blocks see aligned token rows and checkpoint resume is deterministic.

For each atomic block, only `fit` rows determine position-conditioned
median/MAD scaling and the PCA subspace. Independent `cal` rows determine ECDF
references for robust coordinate-tail magnitude and PCA reconstruction
residual. Their calibrated maximum becomes the atomic anomaly score.

Atomic scores are then combined by a fixed hierarchy. Every maximum is itself
recalibrated on the independent calibration stream before the next fusion, so
a family with more coordinates does not win merely because it has more chances
to produce a large score. The primary `full_signal` combines marginals,
balance/scale, retained support, and exact causal topology.

## 6. Fixed evaluation questions

The report gives paired sample-bootstrap intervals for four predeclared
comparisons:

1. `full_signal` versus `attention_marginals` — does the complete method add
   information beyond radial attention mass?
2. `causal_topology_exact` versus `attention_marginals` — is topology itself
   useful without the other families?
3. `rr_multihop_exact` versus `rr_multihop_lag_rewired` — does exact source
   arrangement matter after preserving lag scale and weights?
4. `rr_multihop_exact` versus `rr_one_hop_exact` — does the second hop add
   information beyond local neighborhoods?

The label-free artifact stores scalar scores, token metadata, and two compact
score coordinates only. It does not write population-sized 1024-D/3072-D node
matrices.
