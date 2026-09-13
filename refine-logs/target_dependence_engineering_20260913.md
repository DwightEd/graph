# Target-dependence operator engineering review — 2026-09-13

Reviewed only the new CPU/GPU-capable `route_graph/target_dependence.py` and its
existing native backend contract. This module measures an observed generated
target span under explicit interventions. It is not a correctness decision, a
source-attribution decision, error adoption, or a v3 result.

## Result

**Critical 0, Required 3.**

### Required

1. **A non-finite captured V can be persisted before it is rejected.** In
   `TargetOracle.origin()`, the donor pass returns `CapturedValues` and the
   optional `artifact_writer` receives them before the recipient call
   (`target_dependence.py:118-127`). The recipient backend eventually rejects a
   non-finite donor, but an artifact writer may have already published those
   invalid V bytes. Validate that capture contains every requested layer, the
   exact ordered key shape, floating dtype, and finite values immediately after
   the donor forward and before both artifact publication and recipient-gate
   construction.

2. **A donor/recipient origin intervention can consume one forward and then
   fail on budget exhaustion.** `origin()` performs its donor forward before
   calling `measure()` for the recipient (`target_dependence.py:120,132`). If
   only one call remains, the donor is measured and may be written as an
   artifact, then recipient execution raises `target_forward_budget_exhausted`.
   This violates the stated two-forward operation boundary and leaves a partial
   artifact without a target-dependence result. Validate budget as a positive
   integer at construction and reserve the required uncached forward budget
   before beginning the origin pair (while continuing to count actual forwards,
   including failures).

3. **Query bisection does not enforce interval input and silently loses
   causally invisible children.** `localize_queries()` slices whatever tuple it
   receives (`target_dependence.py:151-165`), although `NativeGate` allows
   unordered/gapped query tuples. It also drops a child when its maximum query
   precedes all keys rather than retaining an explicit
   `causally_invisible`/unsearched record. The output therefore cannot prove
   which query positions were measured, skipped by budget, or excluded as
   future-only. Require sorted contiguous query intervals for this API and
   retain every split child with an explicit status; retain the parent result as
   already intended.

## Confirmed behavior

- `observed_target()` takes a complete contiguous response token event,
  records boundary-overlap token spans, supplies inputs only through the final
  target prediction position, and sums the finite conditional log probabilities
  of every target token with `math.fsum`. Future response tokens are not model
  inputs.
- Baseline and gate measurements check exact unaffected-logit prefixes. A
  strength-one content/route or branch-matched donor sham hashes exact logits
  against baseline and fails closed if it differs. Cached measurements do not
  increment actual-forward counters.
- The donor path applies the raw embedding scaling in the real forward,
  captures selected current V values, binds the donor to input IDs, layer,
  ordered keys, model identity, scaling record, dtype, and values digest, then
  injects it through the receiving attention computation. Native backend GQA,
  causal-mask, and recipient finite-donor checks remain active.
- Per-forward pre-hooks account actual calls and processed input tokens. The
  localization result explicitly says bounded query ranking is neither a unique
  lookback result nor complete when pending intervals remain.

## Verification scope

No `test_target_dependence.py` was present when this review began, and I did
not run a GPU or modify implementation. The parent was concurrently preparing
the dedicated CPU tests, so this initial report is a static contract review.

The still-running population process is unrelated to this unintegrated module.
