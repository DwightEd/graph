# Causal contrast engineering review — 2026-09-13

Reviewed `route_graph/causal_contrast.py` and `tests/test_causal_contrast.py`
against the finite-event and E/V-gate contract. This was a two-pass review:
an independent numerical-contract pass plus a local correctness, integration,
memory, and test review. The module is correctly scoped as a local kernel; it
is not assessed as a complete detector or as evidence from a GPU study.

## Result

**Do not merge until the Required findings are fixed.** There are no Critical
findings.

### Critical

None.

### Required

1. **Stable event contrast can return the wrong finite answer, and can raise
   on finite log-probabilities** — `route_graph/causal_contrast.py:78-83`.
   The alternatives are stabilized before their `logsumexp`, but the final
   subtraction is not. With three one-token branches whose log-probabilities
   are all `-1e308`, `score()` returns `0.0`; the specified value is
   `-log(2)`. The `maximum + log(2)` increment is rounded away before the
   subtraction. Two-token rows of `[-1e308, -1e308]` instead make `math.fsum`
   raise `OverflowError`, despite each accepted token logp being finite.
   Compute the result as `total_B - maximum - log(sum(exp(total_A - maximum)))`
   so the common scale is cancelled before the final arithmetic. Define and
   test the aggregate-overflow policy as well: retain enough precision for an
   accepted finite input domain, or reject an unrepresentable total with the
   documented `ValueError`, never a raw overflow exception. Add extreme-logp
   regression tests alongside the ordinary multi-token/multi-alternative test.

2. **Finite `float64` A/V inputs may silently produce a non-finite gate
   delta** — `route_graph/causal_contrast.py:118-123, 145-146`.
   Inputs are validated while still `float64`, then both are unconditionally
   narrowed to `float32`. For example, `A=[[[1e-50, .5]]]` and
   `V=[[[1e50]], [[1.]]]` are finite float64 tensors; both `content` and
   `route` at zero strength return a float32 `NaN`, where the content delta is
   exactly `-1`. This violates the finite-parameter contract and also makes
   the public result dtype/precision implicit. Keep a safe promoted dtype (at
   least preserve float64), or reject dtypes whose finite range cannot be
   supported before calculation; then validate the produced delta is finite.
   Test finite extreme float64 A/V, and actual `NaN`/`+/-inf` input rejection
   for both attention and values.

3. **The role-mass precondition admits invalid attention** —
   `route_graph/causal_contrast.py:124-126`.
   A role is a slice of a softmax attention row, so its mass cannot exceed the
   receiving row's total mass. The fixed `1.01` allowance accepts e.g.
   `[1.005, 0]`, and route then faithfully preserves the invalid `1.005` mass.
   Use a dtype-aware, near-machine-precision tolerance (or a strict `<= 1`
   after choosing the computation dtype) and report the invariant in the error
   message. Add acceptance/rejection boundary tests.

4. **The GQA contraction needlessly materializes repeated values** —
   `route_graph/causal_contrast.py:145-146`.
   `repeat_interleave` allocates `[K, H, D]`, multiplying the value storage by
   `H / KVH`. The intended integration extracts only a selected query block
   and role keys precisely to avoid retaining large attention-derived objects;
   this allocation can dominate that working set for realistic GQA. Reshape
   heads into `[KVH, repeats]` (or index each query head's KV head) and
   contract against the unexpanded `[K, KVH, D]` values. Preserve the current
   contiguous GQA mapping and add a small equivalence test against the current
   reference computation; a peak-allocation regression check is desirable if
   the test environment can measure CPU allocations.

### Optional

1. **Make the hook-facing tensor contract explicit** —
   `route_graph/causal_contrast.py:86-97, 145-147`. The output order
   `[query, head, head_dim]` matches the existing `ownership` replay path,
   where a caller can reshape it into an `o_proj` input. The function is
   intentionally only a core and therefore does not need to add a model hook
   in this change. Document that it is a single-batch selected-query slice,
   state its returned dtype after fixing the float64 issue, and add a small
   hook-shaped integration test showing that a strength-one delta leaves a
   reconstructed context bitwise unchanged. This would turn the stated
   exact-sham promise into an enforceable interface contract without claiming
   a full model replay.

2. **Add direct validation coverage for the public boundary.** Current tests
   cover the main prefix event, duplicate/prefix rejection, a four-head/two-KV
   GQA mapping, route mass preservation, zero role mass, and an exact-zero
   core sham. They do not cover a nonempty shared continuation beyond the
   prompt separately from its query range, `strength=.5`, empty selections,
   invalid shapes/masks/strengths, or non-finite inputs. The Required cases
   should take priority; these additions would prevent future API drift.

## Confirmed behavior

- `from_sequences()` correctly rejects duplicate and prefix-overlapping
  autoregressive events, requires a shared full prompt, derives the maximum
  common prefix, and scores every post-prefix token rather than averaging by
  length.
- `shared_queries` correctly spans the prompt's last predictor through the
  predictor of the first divergent token, inclusive.
- Route gating renormalizes only its supplied role and rejects removing all
  remaining mass from a nonzero role. A zero-mass role is valid. Content
  gating changes only selected A·V terms. The current GQA head ordering is
  correct.
- A strength-one core gate returns exact zeros. Full-logit equality remains
  the responsibility of the future replay hook, which must recompute the
  downstream network as stated in the module docstring.
- No security concern or population-module coupling was found. The module is
  small, readable, and its stated limitation as a non-detector is accurate.

## Verification

Ran on CPU with the requested interpreter and four OpenMP/OpenBLAS/MKL threads:

```text
python -m pytest -q tests/test_causal_contrast.py
7 passed in 4.81s
```

Manual regressions reproduced the two numerical defects above: the extreme
contrast returned `0.0` rather than `-0.693147...`; finite float64 A/V produced
`NaN` deltas; and a role mass of `1.005` was accepted. `ruff` is not installed
in the supplied environment, so static linting was not run.

## Re-review after the four Required fixes

Reviewed the revised `causal_contrast.py` and its three new regression tests.
The first-round findings are retained above as the record of the defects; this
section assesses the revised implementation only.

**Result: Critical 0, Required 0.** The four prior Required findings are
resolved.

1. `score()` now subtracts the alternative maximum before the log-sum term
   (`:84-89`). The prior three-way `-1e308` reproduction returns
   `-0.6931471805599453`, and an unrepresentable finite two-token total now
   raises the promised `ValueError` rather than leaking `OverflowError`.
2. The gate chooses a promoted working dtype, preserving float64 and promoting
   fp16/bf16 to float32 (`:129-132`), then rejects a non-finite output
   (`:163-164`). The prior finite float64 reproduction now returns an exact,
   finite float64 `-1` delta for both content and route gates.
3. The fixed 1% allowance has been replaced with a dtype-aware tolerance
   (`:134-138`). The former `1.005` float32 row is now rejected, while normal
   low-precision input has an explicit rounding allowance.
4. The GQA contraction groups receiving heads by KV head and contracts directly
   with unexpanded values (`:160-162`). It has no `repeat_interleave` value
   allocation. Independent float64 comparisons against the former expanded
   reference matched for route/content cases with 4/2, 8/2, and 6/3 attention
   heads/KV heads.

The hook-interface documentation/test suggestion remains Optional: this is
still deliberately a local, single-batch kernel, and no end-to-end replay is
being claimed or required here.

### Re-review verification

```text
python -m pytest -q tests/test_causal_contrast.py
10 passed in 6.73s

/tmp/research_lint_20260912/bin/ruff check route_graph/causal_contrast.py tests/test_causal_contrast.py
All checks passed!
```

All commands used the requested CPU interpreter and four OpenMP/OpenBLAS/MKL
threads. No GPU work or code changes were made during this re-review.
