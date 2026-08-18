# Causal attention SetWalk

This is an independent, label-free representation experiment for the two
mechanism hypotheses:

1. hallucination tokens collapse too early onto a small, recent set of
   response-history routes;
2. hallucination tokens do not exhibit the normal cross-layer and cross-head
   coordination between prompt (RP) and response-history (RR) routing.

It does not consume the old RR spectral reference and does not write into an
old topology output directory.

## What is borrowed from CAT-Walk

[CAT-Walk (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/6739d8df16b5bce3587ca5f18662a6aa-Paper-Conference.pdf)
represents a temporal hypergraph as a sequence of set-valued hyperedges. Its
SetWalk moves backward in time through intersecting hyperedges while retaining
the complete member set at every step. The
[official implementation](https://github.com/ubc-systopia/CATWalk) learns
SetMixer and MLP-Mixer parameters for future-hyperedge prediction.

We do **not** copy that detector. Its old training stack, learned SetMixer, and
binary future-edge objective violate this project's no-backprop requirement and
do not match token anomaly detection. We transfer the research paradigm as
follows:

| CAT-Walk | This experiment |
|---|---|
| event-time hyperedge | one `(response token, layer, head)` attention source set |
| earlier intersecting hyperedge | previous-layer hyperedge queried by a current RR source token |
| sampled SetWalk | exact expectation of all retained one- and two-step causal walks |
| node/hyperedge anonymization | prompt-relative position and response causal lag; token IDs removed |
| learned SetMixer | fixed characteristic-function set kernel; no learned weights |
| future hyperedge prediction | train-only one-class reference for each token embedding |

## Dynamic attention hypergraph

For response token `t`, layer `l`, and head `h`, define the weighted hyperedge

```text
E[t,l,h] = {(source s, retained attention A[l,h,t,s]) : s < t}.
```

Prompt tokens are terminal evidence anchors. A response source `j<t` connects
`E[t,l,h]` to every previous-layer head hyperedge queried by `j`. Therefore a
two-step path has the form

```text
E[t,l,*] -> E[j,l-1,*] -> E[k,l-2,*],  k < j < t.
```

This is genuine non-adjacent information provenance across Transformer layers;
it is not `A_l @ A_l` and it never fabricates unavailable prompt-query rows.
Only retained CSR edges are observations. Missing entries remain censored below
the cache floor.

## Fixed set encoding

Every source is anonymized as

```text
x(s|t) = [role, prompt-position-or-log-lag, normalized-source-position].
```

For fixed random frequencies `omega_k`, a complete hyperedge is encoded by

```text
z_k(E) = sum_s normalized_weight(s) * exp(i * omega_k^T x(s|t)).
```

The representation retains `Re(z_k)`, `Im(z_k)`, and `|z_k|^2`. The power term
contains pairwise source-source interactions inside the same hyperedge, while
the characteristic-function samples distinguish multi-modal source sets that
have identical mean lag or identical RP/RR mass. Retained mass, set size, RP/RR
fractions, recent-RR fraction, and weight concentration are appended so that
censoring and scale are not hidden by normalization.

This encoding is permutation invariant over sources and heads but does not
average away the source distribution.

## Exact causal SetWalk propagation

Let `B[t,l,h]` be the full set encoding and `P[t,l]` its permutation-invariant
head-set mean. With transition probability equal to retained RR attention
divided by total retained RP+RR mass,

```text
M1[t,l,h] = sum_j p(l,h,t->j) P[j,l-1]
M2[t,l,h] = sum_j p(l,h,t->j) survival1[j,l-1] M1[j,l-1].
```

The probability is not renormalized over RR alone. Prompt routing therefore
terminates a response-history walk, and the one/two-hop survival mass measures
how much routing remains inside a self-generated response chain. The complete
per-layer state contains:

```text
head_mean(B), head_mean(M1), head_mean(M2),
head_disagreement(B/M1/M2), hop1_survival, hop2_survival.
```

The first three fixed DCT coefficients over the ordered layer trajectory form
the final token representation. DCT is deterministic and preserves low-order
depth dynamics; no labels, gradients, or test-set fitting are involved.

## Required structural controls

All four views use the same train/test samples and the same one-class scoring
protocol:

- `setwalk`: complete set kernel plus two-hop causal walks;
- `no_walk`: complete hyperedge sets without inter-layer propagation;
- `pairwise_walk`: the same walks after replacing each source set by pairwise
  first moments and simple marginals;
- `layer_shuffled`: complete SetWalk with a fixed shuffled layer order.

The claim is supported only if `setwalk` improves over both `no_walk` and
`pairwise_walk`, and its advantage disappears or changes under
`layer_shuffled`. Paired confidence intervals resample complete responses, not
individual tokens.

## Label discipline

The pipeline is strictly staged:

```text
train attention -> fit robust shrinkage reference       no labels
test attention  -> freeze all embeddings and scores     no labels
frozen artifact -> post-hoc token evaluation             labels opened
```

The one-class score uses task/relative-position robust scaling followed by a
contamination-trimmed Ledoit-Wolf precision matrix. This scorer is identical in
principle for every view; it is not a supervised probe.

## Run

Smoke test:

```bash
LIMIT=5 CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/causal_attention_setwalk/run.sh
```

Full experiment:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda \
  bash experiments/causal_attention_setwalk/run.sh
```

Every run receives a new UTC timestamp, so an old result cannot be silently
reused. The terminal prints the exact directory containing:

```text
reference.npz                  unlabeled train reference
nodes.npz                      every token representation and ablation score
evaluation/evaluation.json     full post-hoc report
evaluation/metrics.csv         overall and task-specific view metrics
```

The primary decision fields are
`structural_comparisons.setwalk_vs_no_walk`,
`setwalk_vs_pairwise_walk`, and `setwalk_vs_layer_shuffled`. The scalar
diagnostics are interpretations of the learned-free representation, not the
main method.

