# RR attention topology dynamics audit

This directory studies *why* hallucination tokens leave the RR spectral
subspace. It is a mechanism audit, not a replacement detector chosen after
seeing the test labels.

## 1. Relation to the hidden-state trajectory project

The `DwightEd/demo` hidden-state mainline studies ordered state updates. Its
current hypothesis is not simply that correct reasoning becomes concentrated.
A wrong chain can be internally coherent and converge to a wrong attractor; the
important question is whether a trajectory follows the conditional class of
correct ordered flows.

The attention analogue is:

```text
hidden state H_t       = what internal state the model occupies
attention graph A_t    = how information is routed to form/update that state
```

We therefore test whether response-history routing evolves from many modes to a
smaller, stable set of modes, but **convergence alone is not correctness**. Two
converged regimes must be distinguished:

```text
prompt-grounded convergence:
  prompt evidence -> grounded response relays -> current token

self-reinforcing convergence:
  unsupported response history -> repeated RR feedback -> current token
```

The current attention cache contains no logits, NLL, entropy, or calibrated
confidence. This experiment cannot claim that a lower route rank means the
answer becomes more confident. It only measures routing concentration,
stability, grounding, and topology.

## 2. Data boundary

Every raw attention access goes through `open_research_dataset()` and
`ResearchSample.iter_sparse_attention_blocks()`. No experiment file parses NPZ
or PT caches directly.

A retained RR edge is

```text
(source response token j, target response token t, layer, head, weight), j < t.
```

Missing CSR entries remain censored below `attention_floor`; they are not exact
zero-weight observations. Prompt query rows are unavailable, so prompt tokens
are source anchors rather than a fabricated full prompt graph.

## 3. Frozen RR spectral reference

The audit consumes the previously fitted, label-free RR spectral reference. For
channel `c=(layer,head)`, prefix `t`, and response source `j`,

```text
d[c,t,j]      = sum_{u=j..t} A_c[u,j] / (t-j+1)
lambda[c,t,j] = d[c,t,j] - A_c[j,j]
```

The causal Laplacian is triangular. Its diagonal is its spectrum, and the
strongest-magnitude signed values are retained per channel. The new
`prefix_laplacian_modes()` interface additionally preserves the selected source
index and lag. This is necessary to trace a PCA residual back to actual
response-history anchors instead of reporting only a 5120-dimensional scalar
error.

## 4. Route convergence object

For every token and exact layer/head channel, RR weights are accumulated into
log2 lag bins:

```text
R_t[c,b] = sum_{j<t, floor(log2(t-j))=b} A_c[t,j].
```

Each active channel is normalized over lag bins. Let `P_t` be the resulting
`[channel, lag_bin]` matrix. The audit reports:

- effective rank, participation rank, and stable rank of `P_t`;
- spectral entropy and leading-mode energy share;
- cross-head cosine consensus;
- fraction of active channels;
- cosine velocity between consecutive route states;
- offline distance to the final route state.

A falling effective rank with rising consensus means many channels are using a
smaller family of lag-routing patterns. It does **not** establish correctness.
The final-state distance is explicitly offline/future-using and is never a
causal detector feature.

## 5. Exact-source convergence

Lag-bin rank can hide source identity. Therefore all retained RR weights are
also pooled by exact response source. The audit reports:

- effective number and normalized entropy of active sources;
- largest-source share;
- weighted mean and standard deviation of lag;
- far-history mass fraction;
- source-distribution velocity;
- turnover of the top response anchors.

These features separate `many lag modes` from `many exact source anchors`.

## 6. Prompt-grounded versus self-reinforcing convergence

Prompt support is propagated causally through response history. For token `t`,

```text
direct_t = prompt_mass_t / (prompt_mass_t + RR_mass_t)
relay_t  = sum_j p_RR(t,j) * grounded_j
grounded_t = direct_t + (1-direct_t) * relay_t
```

The complementary relay through low-groundedness sources is reported as
`ungrounded_rr_feedback`. This is a bounded diagnostic of prompt-rooted
provenance under the retained attention graph, not a factuality oracle.

It lets the experiment distinguish:

```text
low route rank + high grounded relay      (potentially supported convergence)
low route rank + high ungrounded feedback (potential self-reinforcing attractor)
```

## 7. Where spectral-subspace reconstruction fails

The frozen PCA residual is reshaped back to

```text
[token, layer, head, selected spectral rank].
```

The audit reports:

- layer-wise residual energy;
- residual energy by selected spectral rank;
- effective number/entropy of abnormal channels;
- top-1 and top-5%-channel residual shares;
- residual-weighted source lag;
- recent/middle/far residual shares;
- residual-weighted prompt groundedness;
- effective number and largest share of implicated source tokens.

This separates three explanations:

1. one isolated head spikes;
2. a distributed set of heads drifts;
3. the same spectral magnitude moves to different source/lag/grounding roles.

## 8. Label discipline and evaluation

The runner has three stages:

```text
train attention + frozen spectral reference
    -> task/position robust topology reference       labels never opened

test attention
    -> raw/z topology trajectories and profiles      labels never opened

frozen feature artifact
    -> post-hoc correct/error and onset analysis      labels opened here only
```

The evaluation writes:

```text
report.json
feature_metrics.csv
within_sample_effects.csv
onset_effects.csv
phase_curves.csv
layer_metrics.csv
spectral_rank_metrics.csv
residual_correlations.csv
```

Feature metrics report both signed direction and orientation-free separability.
Within-sample and first-hallucination-onset effects use sample-cluster bootstrap.
No feature direction or weight is fed back into representation construction.

## 9. Run

The full RR spectral reference must already exist at the default location, or
`SPECTRAL_REFERENCE` must point to another frozen `reference.npz`.

Smoke test:

```bash
LIMIT=5 CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/rr_topology_dynamics/run.sh
```

Full audit:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/rr_topology_dynamics/run.sh
```

Outputs are isolated:

```text
experiments/rr_topology_dynamics/outputs/smoke_5/
experiments/rr_topology_dynamics/outputs/full/
```

## 10. Claim boundary

A lower route rank, higher head consensus, or smaller distance to a final route
state is only evidence of routing convergence. The scientifically stronger
claim requires the following joint pattern:

1. convergence differs at hallucination onset after position/task adjustment;
2. grounded relay and ungrounded feedback distinguish correct/error regimes;
3. residual source/lag/layer attribution identifies a repeatable topology
   change rather than only a scalar mass shift;
4. exact topology later outperforms lag/mass-preserving rewired controls.

The present audit addresses the first three questions. The existing causal
rewiring experiment in `docs/method.md` provides the complementary topology
control, but its intervention does not preserve exact lag, in-degree, or source
collisions and must be interpreted accordingly.
