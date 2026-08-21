# Design audit

## Why the exact-source task was downgraded

The previous CaSH variants tried to discriminate real and rewired source memory,
then to predict exact source identity. The first objective saturated and the
second produced near-random hallucination ranking. These results reject the
claim that exact-source prediction NLL is itself a hallucination score.

The old pipeline remains available as a negative baseline. It is no longer the
main scientific method.

## Active hypothesis

The active experiment tests three propositions separately:

1. weak hallucination information is distributed across the full
   layer-head-source field rather than a single scalar statistic;
2. not every retained attention edge is equally useful for preserving
   source-reuse and prompt-origin structure;
3. a token can be internally easy to reconstruct yet rely more on
   response-origin than prompt-grounded paths.

The third proposition avoids the unsupported assumption that hallucination must
be a low-density or hard-to-predict routing event.

## Observable graph

Only information available in the attention cache is used:

```text
exact source and target token
layer and head
retained attention weight
self attention diagonal
prompt/response role
causal token position
```

No hidden states, logits, values, output projections, or hallucination labels are
used during training or scoring.

## Prompt provenance

Prompt tokens are unit prompt-origin seeds. For a response token at layer `l`,
retained prompt attention contributes directly, retained response attention
inherits the source token's provenance from depth `l-1`, and diagonal attention
retains the current token's previous-depth provenance. Unresolved mass is
ignored, so the result is a lower bound.

This is an analysis operator, not a claim that raw attention equals functional
contribution.

## Label-free objectives

The unmodified graph provides three frozen targets:

- strict-causal received-support top-k field;
- direct-prompt / grounded-response / unsupported-response field;
- prompt-provenance depth trajectory.

Random incidence masking prevents pure copying. The self-supervised loss is the
weighted sum of robust reconstruction losses for these targets.

## Edge refinement

A raw pass computes

```text
S_e = |A_e * d L_self / d A_e|
```

for each retained incidence. `S_e` is detached. A soft gate receives the pair's
layer/head embedding, origin score, predictive sensitivity, mass, and relation.
The refined pass is trained on the same self-supervised targets. Gate-density
regularization is intentionally weak and never uses labels.

Permitted claim if this succeeds:

> label-free predictive sensitivity helps select attention relations needed to
> preserve grounding-relevant graph structure.

It does not identify the language model's true causal circuit.

## Counterfactual interventions

Every response edge receives a continuous prompt-origin coefficient. The frozen
encoder is rerun after removing prompt-origin or response-origin message mass.
Scores are changes in the same self-supervised reconstruction loss:

```text
prompt_gain
response_gain
closure = response_gain - prompt_gain
fragility under small mass-preserving perturbations
```

This is within-token sufficiency analysis, not population-density anomaly
scoring.

## Required controls

The initial implementation exposes raw/refined reconstruction, no-source-state,
matched state shuffle, matched endpoint rewire, and reuse-memory on/off controls.
Before a confirmatory graph claim, it must additionally run layer-order shuffle,
prompt-origin-target ablation, multiple training seeds, and the frozen
`received_topk.causal` baseline under the same token subset.

## Failure criteria

The active method is rejected or downgraded when:

- refined validation loss is not lower than raw validation loss;
- exact endpoint rewiring does not change the graph representation or scores;
- graph compression loses the signal present in `received_topk.causal`;
- closure/fragility are dominated by causal position or unresolved mass;
- counterfactual scores have negligible variance or do not replicate by task;
- performance gains do not survive source-level paired confidence intervals.

A learned SetWalk, larger GNN, or additional topology statistic is not added
until the refined one-hop graph clears these gates.
