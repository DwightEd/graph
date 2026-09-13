# Ownership engineering review, round 2 — 2026-09-12

Separate engineering-review agent; continued application of
`/root/.agents/skills/code-review-and-quality/SKILL.md`. This file preserves the
first-round review rather than replacing it. Scope: updated `ownership.py` and
`test_ownership.py`; new `ownership_multilayer.py`, its shell launcher, and
`OWNERSHIP_MULTILAYER_PLAN_20260912.md`.

This is an engineering review only. No independent scientific approval is
available; `REVIEW_UNAVAILABLE` remains appropriate. This reviewer performed CPU
checks only, and did not edit implementation files or experiment outputs.

## Verdict

The first-round hook-cleanup defect is fixed and independently verified. The new
multilayer operators, causal indices, and fixed 98-condition enumeration are
consistent with the exploratory plan. Before GPU execution, complete the promised
per-query top-1 tie recording identified below. No additional correctness blocker
was found in the intervention implementation.

## Required

**Record top-1 tie counts for every read query.** The plan promises native
argmax/tie information for each query, but `ownership_multilayer.py` initially
records only `native_argmax_ids` for all 132 queries; `max_tie_count` is recorded
only at the two endpoints and the clean control. Save a 132-element tie-count
vector beside the argmax IDs, computed by comparing each logit row with its own
maximum. Since patch raw logits are saved only at endpoints, missing intermediate
tie counts cannot be recovered from those raw artifacts later. This is a bounded
reporting fix and does not change the 98 planned interventions.

The first-round B1 full-vocabulary JS reporting request remains assigned to the
parent's analysis work using persisted world logits. It does not require new GPU
inference and is not re-reviewed as complete here.

## Optional

- Refresh inherited runtime identity fields (`torch`, `transformers`, `gpu`,
  model file metadata), or explicitly label them as inherited from the preceding
  run. The new driver freshly hashes its code and inputs but loads these runtime
  fields from the preceding settings. They may match the unchanged environment
  today, but should not silently describe a future execution after an environment
  change.
- Preserve the dynamic multilayer check below as a small regression test if this
  API will be developed further. Existing tests cover endpoint behavior at one
  layer; the independent check specifically exercises an endpoint swap after an
  upstream layer has already changed the current attention.

## Fixed issue and independent CPU verification

Both capture and intervention registration now lie within their `try/finally`
cleanup boundaries. The new test covers an invalid second patch, an invalid later
capture layer, and a hook-time donor-shape error; it checks no hooks remain and
unpatched logits are restored.

Independently executed a fresh CPU check with a three-layer, four-attention-head,
two-KV-head random Llama in eager mode:

- Same-world XE across all three layers: exact equality for every read logit.
- Same-world MLP across all three layers: exact equality for every read logit.
- Strong X at layer 0 followed by `endpoint_swap` at layer 1: the receiving A
  passed to the second intervention differs from the unpatched layer-1 A,
  demonstrating use of the current forward rather than a stale baseline.
- The endpoint donor A equals the permutation of the current A's two source
  columns; donor V equals current receiving V. Maximum source-mass error was 0
  in this check. The inspected implementation clones the full current A and
  assigns only the selected columns before the common mass-preserving operation.
- All earlier queried logits remained exactly unchanged.
- Hook count was 0 after the successful multilayer call.
- Repeated the original invalid-second-patch witness: hook count remained 0 and a
  subsequent unpatched forward exactly matched the baseline.

These are implementation checks on a small CPU model, not evidence that the
production model will show a desired mechanism. Production bf16 shams and
reproduction checks remain required and are enforced in the driver. The parent's
updated test suite was running at review time; this reviewer did not rerun it.

## Multilayer and indexing audit

Each layer's closures own a separate cache of that forward's `v_proj` values and
pre-O context. The attention hook reads that layer's current A, so downstream
layers account for earlier interventions. Donor A/V/MLP tensors remain fixed to
the declared source world. Endpoint swaps ignore those donor fields, copy current
A, swap the two source-list indices corresponding to absolute positions 76/341,
and keep current V. All directly patched queries follow the source positions, so
causality keeps source-token V unchanged under these response-query patches.

With prompt length 535, t0 reads query 534, grill t99 reads 633, and onion t131
reads 665. `history` scope includes the final prompt query that predicts the first
answer token, as the plan explicitly states. `chosen` indexes prediction steps;
the helper receives their corresponding absolute queries. Comparing
`logits[:min(chosen)]` checks every earlier observed answer prediction. For
history-wide interventions no such earlier answer query exists; reporting `None`
is appropriate. Input tokens remain the fixed observed response throughout, so
these are teacher-forced interventions, not newly sampled trajectories.

Enumeration is 48 layer-group/window/scope/factor conditions + 4 shams + 40
only-one/leave-one-out conditions + 6 endpoint conditions = 98. Each record names
its layers, factor, and exact prediction steps. No outcome-dependent selection or
early success stop appears. The source worlds are re-run and checked for exact
agreement at all 20 previously read queries before local interventions proceed.

## Raw artifacts, provenance, and claim boundary

The driver saves both source worlds' complete 132-query vocabulary logits and
actual edited token arrays, all-layer source factors, each condition's two
endpoint vocabulary logits, and both clean-control vocabulary logits. Each
condition also saves 132-query JS, observed-token log-probability change, entropy,
and native argmax IDs. The tie-count issue above is the remaining predeclared
trajectory-field gap.

The executed-code snapshots now include `route_graph/adoption.py`, both ownership
drivers, the ownership operator, shared I/O helper, and launcher. Input hashes
include both sample/state traces, sample records/settings, the prior manifest,
and the frozen exploratory plan. Fresh output directories and final manifests
preserve previous raw results. The runner remains local/offline and bounded at
four CPU threads; no dependencies or external services were added.

The new plan explicitly acknowledges that it follows the first-round results,
labels the study exploratory, separates final-query/window/history mechanisms,
and distinguishes known-endpoint stress tests from unlabeled detection. It does
not label 14 as a factual repair for the onion step or claim unique-node truth,
classification accuracy, p-values, or cross-source generalization. The correct
onion control still reads t126 at absolute query 660. No claim of independent
scientific approval is warranted by this review.
