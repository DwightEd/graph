# Design audit

## Scientific question

Does exact causal source history improve prediction of a token's current
attention endpoints after current layer/head/weight marks, source role, causal
position, source age, usage count, cumulative mass, and last-use gap are
controlled?

This question precedes hallucination detection. The method is rejected if
history does not improve unlabeled held-out endpoint prediction.

## Evidence and uncertainty

The motivation is the strict-causal `received_topk` result in
`rr_signal_audit`: cumulative future use of response sources contains a stronger
signal than current-row entropy and concentration summaries. That result can
also arise from many weak high-dimensional marginals, position, or cache-floor
effects; it does not prove a source-reuse mechanism.

Transformer attention is known to contain repeatable patterns such as induction
heads and substantial cross-layer redundancy, so route predictability is
plausible. It remains an empirical hypothesis for RAGTruth and for this cache.

## Why the old objective was rejected

The first CaSH prototype discriminated a real source context from one endpoint-
rewired context with an unbounded bilinear logit and sigmoid score. A smoke run
produced nearly constant `-1` scores because positives and negatives were too
easy to separate. Contrastive learning can exploit shortcut differences in
augmentations; graph hard negatives can also be false or trivially distinct.
Training loss therefore did not establish useful anomaly ranking.

## Revised self-supervised target

Version 2 masks one true source identity and ranks it among several strict,
matched alternatives. It uses fixed-temperature cosine InfoNCE and exposes raw
NLL and margins. The candidate count is fixed for every admitted pair.

Candidates preserve:

- prompt/response role;
- fine prompt-position or response-lag stratum;
- source-use-count bucket;
- close cumulative mass, last-use gap, and deterministic history norm;
- causal availability.

Other sources used by the current token are excluded to avoid false negatives.
There is no relaxed fallback. Missing candidates reduce coverage and are
reported rather than replaced by an easier task.

## Model ladder

The same endpoint task is fit with:

1. `current`: matched non-neural history statistics and current attention marks;
2. `birth`: current view plus the source state at creation;
3. `dynamic`: birth state plus subsequent consumer updates;
4. `dynamic:shuffled`: score-time control that permutes memory among matched
   candidates while leaving candidate statistics fixed.

A source-reuse claim requires lower unlabeled validation NLL for `dynamic` than
both `current` and `birth`, a positive real-minus-shuffled memory NLL gap, and a
non-collapsed score distribution. Hallucination AUROC/AUPRC are inspected only
after these gates.

## Validation protocol

- fit and validation samples are split by `source_id`;
- checkpoint selection uses validation endpoint NLL, never training loss;
- all position features are prefix-causal;
- score artifacts contain raw NLL, raw cosine logits, margin, candidate count,
  match distance, valid coverage, and embeddings;
- labels are opened only by the final evaluation command.

## Permitted claim

If all gates pass, the permitted claim is:

> Exact source-reuse history improves prediction of attention routes, and route
> prediction error is associated with hallucination.

The experiment does not establish that attention values were adopted by the
residual stream, that the MLP used them, or that the routes caused the output.

## Failure criteria

The method is rejected or downgraded to a negative result when any of the
following holds:

- `dynamic` validation NLL is not lower than `current` and `birth`;
- real memory does not outperform matched shuffled memory;
- valid-token coverage is too low or concentrated only at late positions;
- endpoint NLL has negligible variance or too few unique values;
- detection does not exceed current/birth and the frozen
  `received_topk.causal` baseline under paired source-level confidence intervals.

A learned temporal SetWalk or source-coalition module is not added until the
one-hop dynamic memory clears these gates.
