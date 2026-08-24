# Validation Plan

## H1: Non-Markov path memory

Primary label-free statistics:

- validation `order2_gain`;
- validation `order2_path_gain`;
- validation `order3_gain`;
- validation `order3_path_gain`.

Controls:

- matched-dimension shuffled path blocks;
- exact middle-node order retained only in the real path representation;
- source-disjoint train/validation split.

Decision:

- authorize an order-2 graph only when both order-2 gains are positive across seeds;
- authorize order 3 only when it adds to order 2 and beats its shuffled block.

## H2: Anchor-path congruence

Primary comparisons:

- `anchor_js_peak` versus `direct_role`;
- `anchor_js_excess` versus anchor-ID permutations;
- explicit-anchor manifest versus uniform prompt chunks.

Decision:

- do not use the term evidence grounding unless an evidence-aware manifest beats the prompt-chunk and prompt-role baselines.

## H3: Evidence audit escape

Primary statistic:

- hallucination-minus-matched-correct `evidence_escape` with source-group bootstrap CI.

Matching variables:

- same response;
- causal position;
- known anchor mass;
- response-base mass.

Decision:

- candidate support requires a negative effect and matched `d_z <= -0.20`.

## H4: Response-walk lock-in

Primary statistics:

- lock-in AUPRC above prevalence;
- first-error onset change minus all-correct pseudo-onset change;
- post-onset response persistence and absent evidence escape.

Decision:

- a static token difference without post-onset persistence is not lock-in.

## H5: Base-model causality

Not implemented in this cache-only package. Required future data:

- projected Q/K alignment;
- V and output-projection contributions;
- residual and MLP updates;
- targeted path knockout and evidence restoration;
- matched random interventions and dose-response controls.

## Engineering validation

The tests cover:

1. anchor manifests and fallback chunks;
2. lineage mass conservation and relay-depth shifts;
3. explicit De Bruijn predecessor construction;
4. synthetic order-2 and order-3 predictive gains;
5. direct/relay JS divergence and recoupling;
6. query-to-next-token alignment;
7. score artifact schemas and end-to-end evaluation.
