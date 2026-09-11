# Round 4 Implementation Consistency Review

## Scope

This review checks whether the final research specification in `refine-logs/round-2-refinement.md` is faithfully implemented by:

- `route_graph/operator.py`
- `route_graph/capture.py`
- `route_graph/detector.py`
- `route_graph/evaluation.py`

I did not modify implementation files. I did not require natural performance evidence as a code-change prerequisite. Scientific status remains the same as Round 3: **REVISE, 8.5/10, not paper-ready until source-disjoint natural results exist**.

## Summary

The core implementation matches the final method. I verified:

- predictor index `q = prompt_tokens + token_index - 1` is used, with response suffix present in the teacher-forced input but sliced away from features at `0..q`;
- candidate signal implements top-1 vs other top-K cosine contrast from final logits and normalized output embeddings;
- attention is treated as receiver-first/source-second, with causal row-stochastic validation;
- the null preserves role/source-unit/log-lag/self structure through grouped prefix sums;
- source/history one-step, two-step, and source-through-history paths follow the intended layer order;
- source-disjoint reference scoring is enforced;
- shared-scale active masking implements the Round 3 standardization rule;
- labels are joined only in evaluation, after frozen scores exist.

I found **one substantive reporting gap**. It does not change the mathematical score computation, but it affects whether evaluation reports can be interpreted according to the final specification.

## Finding

### IMPORTANT: evaluation does not report inactive-reference coverage

`RouteDetector` writes `reference_active_features` into every score row at `route_graph/detector.py:239`, and inactive strata are scored as zero by `SourceReference.score()` at `route_graph/detector.py:112-115`. This matches the final method only if downstream evaluation makes clear when a zero score came from no active reference coordinates.

`RouteEvaluator` currently builds the public evaluation report at `route_graph/evaluation.py:146-160`, but that report includes only subsets, paired AUROC gains, generated-token candidate coverage, and scope. It does not summarize:

- how many evaluated tokens had `reference_active_features == 0`;
- the distribution of active feature counts;
- whether each subset's metrics were dominated by inactive-reference tokens.

The final specification explicitly says that a stratum with no active coordinates should get zero scores and be marked, but should **not** be interpreted as credible normal behavior. Because evaluation reports AUROC/AP over all tokens without active-feature coverage, a downstream reader can see zeros inside the metrics without the required applicability warning.

Recommended implementation change, not performed here:

- add `reference_active_features` coverage to the evaluation report, at least total and per subset;
- preferably report metrics for all tokens and coverage for `active_features > 0`, without using labels to filter;
- keep the existing all-token metrics, since the score file itself is frozen and label-free.

## Verified Consistency Points

### q position and no future/target leakage

`capture.py:120` builds `ids = prompt_ids + response_ids[:-1]`; `capture.py:200-207` uses `query = len(prompt_ids) + token_index - 1`, logits at that predictor, and signals sliced to `:query + 1`. `operator.py:42-44` slices attention, signals, and roles to the same prefix. This matches the final definition for first token and later tokens.

### Candidate cosine contrast

`capture.py:201-207` takes top-K from the predictor logits, normalizes the selected output embeddings, builds `directions[0:1] - directions[1:]`, and multiplies by normalized final-norm states. This implements `cos(F(h_l(i)), u_Cq[0]) - cos(F(h_l(i)), u_Cq[k])`.

### Source role and strong null

`capture.py:141-173` assigns source/instruction/history/special roles without using labels and gives source tokens structural paragraph units. `operator.py:52` maps source units into separate source groups, while `operator.py:113-140` applies the null within group and log-lag bucket while preserving self mass. This matches the strong role/source-unit/log-lag null.

### Path direction and source-through-history naming

`operator.py:71-77` propagates earlier layers in true layer order with averaged earlier heads. `operator.py:79-86` masks the intermediate history nodes only for the source-through-history path, and `operator.py:91-101` applies the final head-specific read. This matches the final two-step residual interpretation, not pure interaction.

### Source split and reference scoring

`detector.py:96-103` rejects any scored source that appears in the fit source set. `detector.py:176-217` also checks duplicate tokens, predictor metadata, response-level fixed fields, and complete token streams before fitting references. This satisfies the source-disjoint and complete-capture constraints.

### Shared scale and active mask

`detector.py:76-90` computes the shared per-coordinate scale as the maximum within-view reference standard deviation across residual/observed/null/signal, uses `scale > 1e-6` as the common active mask, and centers each view by its own reference median. `detector.py:112-124` applies the same active mask for score and kNN. This matches the Round 3 standardization rule.

### Evaluation label boundary

`evaluation.py:37-93` joins labels by response only after scores are loaded and verifies response/source digests before assigning token labels. `evaluation.py:100-145` evaluates all tokens, onsets, first errors, continuations, source-balanced metrics, and paired residual-vs-baseline AUROC gains. This matches the label-boundary and primary comparison design.

## Scientific Status

No natural performance evidence is reviewed here. The implementation being consistent does not make the paper claim true. The scientific verdict remains:

- **Score:** 8.5/10
- **Verdict:** REVISE
- **Implementation:** mostly ready, with the active-feature reporting gap above
- **Paper readiness:** not ready until natural source-disjoint residual gains are shown

The result should still be described as an implementation-ready hypothesis test, not as evidence that hallucination detection works.
