# Label-free causal topology representation

This experiment asks whether the *arrangement* of retained causal attention
contains token-level hallucination signal beyond simple attention marginals.
It has no learned message-passing network, and the algorithm does not expose or
consume labels while encoding, fitting references, or producing anomaly
scores. Evaluation labels are supplied only after the score artifact is frozen,
for AUROC/AUPRC and paired-bootstrap evaluation. A serialized cache may still
physically contain an embedded label field; that field is excluded from the
algorithmic data flow rather than claimed never to enter process memory.

## 1. Sparse attention semantics

Let the cache threshold be \(f=0.01\). A missing CSR entry only tells us that
the original edge was below the retention threshold; its exact weight is not
identifiable. The primary representation therefore uses the retained-mass
point estimate rather than assigning every missing edge the upper bound
\(0.01\):

\[
M^P_{t,c}=\sum_{s\in P\cap E_{t,c}}a_{t,s,c},
\qquad
M^R_{t,c}=\sum_{s<t,\,s\in E_{t,c}}a_{t,s,c}.
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

If both retained means and the stored diagonal are zero, the undefined balance
coordinate is filled with \(f=0.01\); this is the requested fallback for an
undefined Lookback value, not a claim that every censored edge equals \(f\).
Retained prompt/history edge fractions are stored separately, so the anomaly
model can represent how strongly a row was censored.

## 2. Prompt provenance without prompt-length pooling

A prompt source position \(s\) is encoded by \(K\) Fourier frequencies:

\[
\phi_K(s)=\left[\sin(2\pi ks/|P|),\cos(2\pi ks/|P|)\right]_{k=1}^{K}.
\]

For every token and layer-head channel, the encoder stores the retained-weight
mean of \(\phi_K(s)\). This preserves coarse and fine prompt-source location
while keeping a fixed \(2K\)-coordinate interface for different prompt
lengths. `FOURIER_FREQUENCIES` controls \(K\); the default is 4.

## 3. Response-history topology

For each response target and channel, retained RR weights are normalized over its
retained history neighbors. The encoder does not mean-pool layers or heads. It
emits, per channel:

- the normalized neighbor mean of the two-coordinate state \(x\);
- the normalized mean absolute target-neighbor difference;
- the normalized neighborhood variance;
- a second-hop neighbor mean obtained by routing the first-hop means once more.

Signed one-hop differences can cancel even when a neighborhood is abnormal.
The absolute difference and variance retain that dispersion, while the second
hop distinguishes similar marginals produced by different causal paths.

## 4. Coarse-lag-stratified topology null

The control graph keeps every RR target, channel, retained weight, causal
validity, and \(\lfloor\log_2(\mathrm{lag})\rfloor\) bin, then draws another
source from that bin when an alternative exists; singleton bins remain
unchanged. Prompt routes and all attention marginals remain unchanged. The
intervention does **not** preserve exact lag, source in-degree,
or source-collision patterns. Exact-versus-rewired performance therefore
measures sensitivity to this joint coarse-lag-stratified source intervention;
it cannot by itself isolate source identity from the induced degree, collision,
and within-bin lag changes.

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

1. `full_signal` versus `attention_marginals`: does the complete method add
   information beyond radial attention mass?
2. `causal_topology_exact` versus `attention_marginals`: is topology itself
   useful without the other families?
3. `rr_multihop_exact` versus `rr_multihop_lag_rewired`: is the score sensitive
   to coarse-lag-stratified source rewiring? This comparison does not isolate
   exact source identity because in-degree, collisions, and exact lag may also
change.

The three CSR passes are streamed in `ROW_BLOCK_SIZE` rows (4096 by default),
so temporary edge tensors scale with one block rather than the largest sample's
complete edge set. Changing the block size does not change the encoding.
4. `rr_multihop_exact` versus `rr_one_hop_exact`: does the second hop add
   information beyond local neighborhoods?

The label-free artifact stores scalar scores, token metadata, and two compact
score coordinates only. It does not write population-sized 1024-D/3072-D node
matrices.
