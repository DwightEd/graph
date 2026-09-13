# Restoration v2 source-dependence engineering review — 2026-09-13

Scope: `next_iteration/grounded_graph_restoration_dependence.py`, with read-only inspection of the frozen v1 source-payload-erasure cache contract and v2 feature/train interfaces. I did not alter implementation, run GPU, or inspect labels.

## Findings

**Critical: 0.**

**Required: 1. Fail closed on nonfinite or out-of-range branch measurements before aggregating or publishing the source48 gate.**

`erased_arrays()` correctly rejects nonfinite cached feature arrays, but the result of `measure()` is written and aggregated without checking `base_logp`, `adapter_logp_gain`, or `coordinate_pointer_probability`. A NaN can silently enter `np.mean`; the comparisons then yield a false gate rather than an explicit invalid diagnostic. Require every per-anchor scalar to be finite, and require coordinate probabilities to be in `[0, 1]` up to a small floating-point tolerance, before recording a source. This makes a malformed adapter result an integrity failure rather than a negative mechanism result.

## Checks that passed

- The two arms use the same all-source-erased `H_empty` query. The payload-erased arm swaps only the actual selected-owner re-encoded source X; cached v1 query states are schema- and byte-checked but deliberately not used.
- The v1 cache is constrained to exactly the 48 distinct source-validation IDs, complete artifacts, 48 actual observer forwards, the original source feature manifest, the same observer model bytes, and its frozen live/snapshot code.
- Each cache record is bound to the exact packet, task/source identity, original receipt, target-owner set, union token positions, replacement ID, executed input IDs, per-array shape/dtype/hash, and finite arrays. This prevents treating a cache from another packet, owner set, or node order as a payload intervention.
- Anchor gains have the intended sign: `adapter_logp - logp_empty`; payload dependence is `gain_original_X - gain_erased_X`. The gate requires both positive source-mean anchor gain and positive fixed-query gain drop. The full-source `H_full` logp is saved separately and does not leak into that gate.
- Training, restoration-feature, and cache parents are transitively sealed through the training manifest, feature verification, original manifest hash, model manifest, and live/snapshot code checks. No labels or native-route claim is introduced.

## Independent CPU tests

Added `tests/test_grounded_graph_restoration_dependence_review.py`:

- validates that the cache supplies only erased X while the v2 `H_empty` query remains the one passed to both branches; a different selected-owner cache rejects;
- validates source-mean aggregation and that a positive erased-X drop cannot pass the gate when original gain is negative.

Focused result: **2 passed**; Ruff is clean.

## Closure re-review

The Required item is **closed**. `validate_branch()` now requires a nonempty complete anchor vector, finite base logp/adapter logp/gain/pointer values for both arms, and pointer probabilities in `[0, 1]`; `run()` applies it to both fixed-`H_empty` branches before baseline comparison or artifact writing. The separately recorded full-source anchor logp is also required finite.

The focused suite now has **3 passed** and Ruff is clean. It includes explicit NaN and out-of-range-pointer rejection. **Final status: Critical 0, Required 0.** The result remains a held-out source-coordinate restoration/content-dependence diagnostic; it does not establish semantic necessity, native routing, or natural-error detection.
