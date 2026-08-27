# Current research status

## Active question

The active experiment asks whether a label-free neural graph encoder benefits
from reconstructing the exact endpoint distribution induced by composing
sparse attention transitions in Transformer layer order.

This is narrower than factual grounding. The available cache contains
attention rows, token boundaries, diagonal mass and unresolved sparse-cache
mass. It does not contain hidden states, per-head OV messages, FFN outputs or
prompt-query rows.

## Active implementation

`experiments/directed_route_hypergraph/` is the current experiment. One sample
produces one causal typed graph:

```text
node: token
edge: (source, response target, layer, head, retained weight)
row: retained + diagonal + unresolved = 1
```

The neural encoder receives a mass-conserving corrupted graph and is trained
against three clean, label-free targets:

1. local retained endpoint + SELF + UNRESOLVED row distributions;
2. layer-ordered prompt/response/unresolved trajectories;
3. a final layer-ordered distribution over token endpoints retained by the
   attention proxy plus an absorbing unresolved sink.

The endpoint objective is factorized into sink mass, self mass conditional on
resolved transport, and log-candidate-normalized non-self endpoint shape. This
reduces the direct unresolved/self shortcut of one categorical loss; it does
not rule out position or length shortcuts.

The encoder still outputs one 64D vector per token. PCA-whitened kNN is fitted
only on source-disjoint unlabeled calibration embeddings. Test labels are
opened only after scores are saved.

## Relation to Information Flow

The active endpoint target uses the same algebraic idea of multiplying
non-commuting layer transitions. It is not the contribution layout from
*Information Flow Reveals When to Trust Language Models*:

- paper edge weights depend on hidden states, `W_V`, `W_O`, residual output and
  ALTI-style vector attribution;
- active edge weights are raw retained attention plus a fixed residual proxy;
- paper features use neural reranker/SHAP relevance and a supervised XGBoost
  calibrator;
- the active encoder is neural and label-free, followed by one-class kNN.

The permitted name is `layer-ordered attention transport endpoint layout`.
The code must not call it functional contribution, causal information flow or
grounding.

## Rejected closure branch

The previous P-Cut closure hypothesis is not active. Its frozen full-QA result
was AUROC `0.4209` and AUPRC `0.0734`, below position baselines. The current
implementation contains no full/no-prompt/no-response cuts, no closure score
and no post-hoc direction reversal.

## Evidence currently available

Implemented synthetic tests establish:

- non-negative row mass and unresolved-sink conservation;
- actual layer-order sensitivity on a non-commuting graph;
- exact prompt endpoint identity;
- future-response and truncated-prefix invariance;
- causal masking in the endpoint pointer decoder;
- balanced layout-loss batching invariance and gradient flow;
- existing typed-graph, corruption and label-boundary invariants.

No full RAGTruth ordered-layout result is recorded. These tests establish
implementation consistency, not detection validity.

## Required ablations

Use the same source split, budget and downstream detector for:

1. local row only;
2. local + P/R/U;
3. local + endpoint layout;
4. all three objectives;
5. correct layer order versus reverse, shuffled and last-layer layouts;
6. real endpoints versus matched endpoint rewire and weight shuffle;
7. full encoder versus position-only and self+unresolved decoders;
8. neural embeddings versus the deterministic endpoint layout itself;
9. residual proxy values `0`, `0.5`, `1`, `2`;
10. a small value-aware cache subset versus attention-only layout.

Report target reconstruction as an unlabeled diagnostic. Select no loss weight,
layer order, score direction or detector hyperparameter using test labels.

## Acceptance gates

A layer-ordered endpoint-flow claim requires all of the following:

1. local + endpoint improves held-out clean-layout reconstruction over local
   only without increasing position predictability;
2. correct order outperforms reverse/shuffled order under matched capacity;
3. real endpoints outperform role/lag/mass-matched rewiring;
4. non-self endpoint loss cannot be matched by position-only or
   self+unresolved controls;
5. token AUPRC improves over GroundedRoute, direct endpoint layout and
   position controls on source-disjoint test groups;
6. source bootstrap and at least five seeds support the gain;
7. QA, Summary and Data2txt are reported separately;
8. a value-aware subset shows that the attention proxy has adequate endpoint
   overlap or faithful route-token deletion effects.

## Stop rules

```text
ordered ~= reverse          -> remove the layer-order contribution claim
real ~= endpoint rewire     -> remove exact-endpoint/topology claims
layout ~= position-only     -> endpoint target is a positional shortcut
layout dominated by sink    -> reject the target or recollect denser caches
attention != value-aware    -> call it rollout only; do not scale the claim
embedding ~= direct layout  -> remove neural graph-encoder novelty claim
no AUPRC gain               -> retain as representation audit, not detector
```

## Next experiment

Run one frozen QA matrix before adding another module:

```text
FLOW_WEIGHT=0   LAYOUT_WEIGHT=0
FLOW_WEIGHT=.5  LAYOUT_WEIGHT=0
FLOW_WEIGHT=0   LAYOUT_WEIGHT=.25
FLOW_WEIGHT=.5  LAYOUT_WEIGHT=.25
```

Archive checkpoints, graph sidecars, embedding indices, scores, evaluation,
commit SHA and the exact source split for every cell. Then run reverse endpoint-target,
endpoint-rewire and position-only controls. Do not infer effectiveness from
the implementation tests alone.
