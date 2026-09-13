# A2-v2 restoration design bounded review — 2026-09-13

Scope: `docs/GROUNDED_GRAPH_RESTORATION_V2_20260913.md` plus the v1 failure facts stated in the task and plan. I did not modify frozen v1 artifacts, run GPU, or inspect labels. This is a focused `research-refine` method review, not a new literature survey.

## Verdict

**Proceed only after the three Required clarifications below are made explicit in the implementation/run doc.** The v2 idea is the right minimal response to the observed v1 failure: v1 learned pointer sensitivity, but generation gain did not depend on source payload and was already negative. Replacing the query and residual decoding base with a real all-source-erased observer state `H_empty` directly removes the easiest bypass where `H_full[t-1]` already carries copied source text.

This remains a restoration model, not an internal-causality proof. It can test whether graph-side original source states help predict observed tokens beyond an empty-source baseline. It still does not solve QA/natural-response semantic ownership, long-range propagation, or original-generator route attribution by construction.

## Required before launch

1. **Define and receipt the all-source-token erasure boundary exactly.**

   “All source token” must mean every prompt token whose tokenizer offset has positive character overlap with the frozen `source_span`, including tokens crossing source-span boundaries. It must exclude BOS, task/question/prompt text outside the source span, response-prefix tokens, and target tokens. Each packet should save:

   - original input id digest;
   - erased input id digest;
   - erased token indices;
   - source-span overlap rule;
   - unchanged non-source prompt and response-prefix check;
   - single replacement token id and token count equality.

   Without this, a partial source erasure can leave a hidden source-copy path, while an overbroad erasure can erase the QA question or response prefix and change the task being measured.

2. **Full-token CE is still exposed to copied-prefix leakage; anchor-token restoration must be reported and used for selection or gating.**

   The current token is not in `H_empty[t-1]`, but previous teacher-forced copied source tokens are. On a source-copy reconstruction sequence, later tokens in a paragraph/value can often be predicted from earlier copied response tokens rather than from graph-side source nodes. With 135,519 target tokens and only 9,220 coordinate anchors, all-token CE can again be dominated by formatting and local continuation.

   Minimal correction: keep full-token CE for training if desired, but predefine separate validation metrics for:

   - first coordinate-anchor token CE/gain;
   - non-anchor CE/gain;
   - pointer NLL/top1;
   - source-payload-erased-X drop at anchor tokens.

   Checkpoint selection should not be allowed to choose a model that improves only non-anchor CE while failing anchor restoration and pointer objectives. If selection remains all-token CE + pointer only, the run can reproduce v1’s failure mode and still pick a misleading checkpoint.

3. **Separate score definitions and units; do not merge full-gap and empty-restoration scores.**

   `logp_full(y) - logp_restore(y)` and `logp_empty(y) - logp_restore(y)` answer different questions. The first compares restoration to the original full-source observer; the second measures whether graph restoration improves over the erased-source observer. Both can be useful fixed scores, but neither is calibrated probability and they should not be averaged or used interchangeably.

   Predefine names such as:

   ```text
   risk_full_gap = logp_full - logp_restore
   risk_empty_restoration = logp_empty - logp_restore
   support_empty_restoration = logp_restore - logp_empty
   ```

   The source-heldout restoration control must first show positive anchor-level `support_empty_restoration` and payload-erasure sensitivity before natural AUROC changes are interpreted as evidence that graph source content is being used.

## Method-specific critique

- The core change is small and well targeted. It keeps the same adapter, source split, graph, pointer objective, and no-edge arm, and only changes the information path available to the query/decoder base.
- The source-copy task is still distribution-shifted from natural QA/Summary answers. `H_empty` may make this shift larger because the erased-source prompt is unnatural. That is acceptable only because the natural full-word/post-first evaluation remains the deciding test.
- For Data2txt, source erasure must include literal value tokens but not the field/record graph X used as source nodes. If X is accidentally recomputed from the erased prompt for the main branch, the restoration objective collapses.
- The planned fixed-query source-payload re-encoding control remains necessary. Since `H_empty` already removes prompt source from the query, the decisive control is whether erasing the graph-side original payload states reduces anchor logp gain and coordinate pointer probability.
- A surviving gain after graph-side erasure is still ambiguous: duplicate source occurrences, response prefix history, and formatting priors can carry the target. The report must preserve the redundant-evidence caveat instead of calling survival “format only.”

## Minimal implementation contract

The smallest implementation that can support the v2 claim is:

1. Build `H_full` and `H_empty` from the exact same packet/tokenization, differing only in prompt source-token replacement.
2. Use `X_original` for source nodes in the main restore branch; use `H_empty[t-1]` for query and residual base.
3. Save branch receipts for original full, all-source-erased query, graph-source-erased-X, no-edge, and permuted-edge variants.
4. Train graph and no-edge arms with matched seed/order/capacity and the same source train/validation SHA split.
5. Select checkpoint using a predeclared validation objective that cannot ignore anchor restoration.
6. Freeze scores before natural labels: LM NLL/entropy, empty-source NLL, restore NLL, `risk_full_gap`, `risk_empty_restoration`, graph/no-edge/permuted differences, and anchor-control drops.

## Decision boundary

If v2 fails to show positive source-heldout anchor restoration and payload-erasure sensitivity, stop this adapter route rather than adding epochs or more graph modules. If it succeeds on source restoration but not natural full-word/post-first detection, the correct conclusion is “source-coordinate restoration works under copy pretraining but does not transfer to natural hallucination detection,” not “internal ownership is solved.”
