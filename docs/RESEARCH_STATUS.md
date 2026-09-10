# Current research status

## Active question

At a factual commitment, does the generated prefix override the source
constraint even when the source still causally affects the answer candidates?

This is more specific than “hallucinations attend locally.” The proposed
failure has three parts:

1. the source can still influence the current choice;
2. the generated prefix independently pushes the follow-up choice;
3. source-prefix coupling is abnormally weak or strong relative to comparable
   normal events, so source control is not stably carried into continuation.

## Implemented method

One event is measured under two source worlds and two generated prefixes. The
six answer-token margins are decomposed into four causal graph edges:

```text
source constraint -> onset choice       source_onset
source constraint -> follow-up choice   source_followup
generated prefix  -> follow-up choice   prefix_followup
(source, prefix)   -> follow-up choice   source_prefix_coupling
```

The graph schema is fixed before labels are read. A robust reference profile is
fit separately for every relation type on the unlabeled fit split. The score is
the mean squared robust z-distance of the four edge weights. The output also
reports each edge's signed standardized deviation, so a high score remains
mechanistically inspectable instead of losing the direction of change.

The fit and score splits must be source-disjoint. A relation without at least
four fit events has no defensible reference distribution and is rejected. The
evaluation module joins hallucination labels only after scores are frozen.

## Why the previous autoencoding direction is stopped

The earlier ordered-layout experiment did not support reconstruction as the
main method. Autoencoder AUROC/AUPRC was `0.540130/0.081410`; Deep SVDD was
`0.526438/0.078837`; position alone was `0.617076/0.112859`. Predicting one
layer from the next is also dominated by residual continuity, token position,
and shared computation. Low prediction error therefore need not mean that a
constraint was correctly bound or used.

GraphMAE shows why reconstruction design matters: it masks attributes and
avoids naive structure reconstruction. GOOD-D and one-class graph
transformation learning show viable label-free graph baselines, but they assume
enough graph diversity to learn a common normal manifold. Our graph currently
has fixed roles and only four causal coordinates; adding a GNN or decoder now
would add parameters without adding identifiable information.

Neural self-supervision is deferred until the deterministic graph beats
position and simple non-graph controls. If used later, the candidate objective
is masked causal-edge recovery or counterfactual equivariance—not
layer-to-layer prediction.

## Preserved empirical clues

The old attention audit supports a sharper candidate than generic “failure to
look back”:

- at hallucination onset, many heads shifted from local history toward distant
  response history while prompt share still decreased;
- inside hallucinated spans, local-history share increased and attention
  change decreased;
- 94.46% of matched hallucinated tokens were not onset tokens;
- signed head messages could cancel, so attention mass alone did not establish
  use of the attended information;
- the supplied reanchor run found candidate anchors in only 23 of 688 samples
  and no successfully traced anchors in the shown records.

The resulting hypothesis is **old-response relay takeover**: the model may
retrieve semantically related remote content but fail to restore the source
condition that governs the correct relation. The prefix then becomes a stable
local attractor and expands the first wrong commitment. These observations are
hypothesis-generating, not yet causal evidence.

## Required experiments

1. Build minimal A/B source worlds and A/B prefixes for RAGTruth factual events.
2. Freeze commitment-token and answer-candidate alignment before reading labels.
3. Fit reference profiles only on source-disjoint unlabeled data.
4. Compare the four-edge score with position, margin, entropy, attention-only,
   Isolation Forest, Deep SVDD, and an autoencoder under identical splits.
5. Report event coverage, relation coverage, AUROC, AUPRC, prevalence, and
   source-cluster bootstrap intervals.
6. Test normal/hallucination edge distributions on a discovery split, but never
   use those labels to change the graph schema or detector on the test split.
7. Replicate across QA, summarization, and data-to-text; at least two model
   families are required for a general mechanism claim.

## Claim gates

- If the score does not beat position and raw-margin controls, it is not a graph
  anomaly result.
- If source or prefix interventions do not move the answer margin, the event is
  non-identifying and must be reported as uncovered, not silently discarded.
- If performance vanishes after relation-conditioned calibration, the original
  effect was task/relation confounding.
- If normal and hallucinated events differ only inside an already-wrong span,
  the signal is a consequence, not an onset detector.
- If an autoencoder only lowers reconstruction loss, no hallucination claim is
  permitted.
- A model-mechanism claim requires consistent directional effects and causal
  interventions across models; AUROC alone is insufficient.

## Current limitation

The repository implements graph construction, label-free scoring, and isolated
evaluation. It does not yet capture the six factorial margins from a language
model. That stage requires an explicit event-construction protocol; adapting
RAGTruth automatically without verified answer candidates and counterfactual
worlds would manufacture invalid ground truth.
