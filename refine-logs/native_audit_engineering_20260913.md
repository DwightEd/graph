# Native audit backend engineering review — 2026-09-13

Reviewed the paired-R extension in `route_graph/causal_contrast.py`, the new
`route_graph/native_audit.py` backend, the two test modules, and the backend
contract in `refine-logs/architecture_round4_20260913.md`. This was a two-pass
review: an independent numerical/provenance pass and a local five-axis review.
This assesses a low-level frozen-Llama measurement backend only; it does not
assess unimplemented semantic anchoring, candidate search, certificate
orchestration, or detector effectiveness.

## Result

**Do not merge until the three Required findings are fixed.** There are no
Critical findings.

### Critical

None.

### Required

1. **A naked donor tensor cannot enforce the required branch-matched V
   provenance** — `route_graph/native_audit.py:25-26, 157-160` and
   `refine-logs/architecture_round4_20260913.md:141-145`.
   `NativeGate.donor` holds only a tensor and validation checks only the shape
   after casting. A capture from another full input length, branch/input-scale
   condition, layer, or key ordering can have the same `[key, KV_head, D]`
   shape and is accepted. I reproduced this by capturing keys `(1, 2)` from a
   six-token branch and applying the tensor to a seven-token recipient; the
   backend ran normally. This invalidates the stated same-original-shape donor
   contract and can turn an unmatched replacement into an alleged mediated
   effect. Return and accept a structured `NativeDonor` capture containing at
   least the values, layer, ordered keys, input token hash/length, KV-head
   count, and source dtype. Validate its layer/key order and the recipient
   branch identity before hooks are installed. The donor forward may differ in
   embedding scale, so that condition should be recorded separately from the
   unscaled input identity rather than compared as the values themselves.
   Add negative tests for wrong input length/IDs, wrong layer, and same-count
   reordered keys.

2. **Donor payload validation permits fabricated non-floating inputs and
   fails late for non-tensors** — `route_graph/native_audit.py:157-160`.
   `torch.ones(values.shape, dtype=torch.int64)` is converted to model dtype
   and accepted as a donor; I reproduced a completed replay with it. An object
   without `.to()` instead fails inside the forward hook as `AttributeError`.
   A branch-captured V must be a finite floating tensor, before any dtype/device
   conversion. Validate `isinstance(..., torch.Tensor)`, floating dtype,
   finiteness, and the structured provenance during the pre-forward gate
   validation; retain the hook-side current-shape check as a defense against
   model-state inconsistency. Add tests for integer, non-tensor, and non-finite
   donors, and verify hook cleanup on each rejected path.

3. **The native gate API accepts keys that are future to a gated query** —
   `route_graph/native_audit.py:73-92, 151-156`.
   The backend knows full token positions yet does not enforce causal
   visibility. A `content` gate for query `2` and key `5` was accepted and
   silently acted on the mask's zero weight. The architecture requires the
   native graph to contain only keys visible at the receiving computation; a
   silently no-op future selection is not a valid tested group. Require each
   key domain to be visible to every selected query (for the present rectangular
   API, `max(keys) <= min(queries)`), or explicitly partition the operation by
   query-visible keys. Apply the same check to selected and destination through
   their key domain, and add route/content/donor future-key rejection tests.

### Optional

1. Add a tiny native paired-R integration test with 4 attention heads and 2 KV
   heads. The kernel test establishes the transfer formula and GQA ordering,
   but an eager-Llama replay test would cover the hook's selected/destination
   position mapping and full-shape O replacement together.

2. Add a direct dynamic-value test: install a lower-layer gate, capture V from
   a later layer, and show it differs from the ungated capture. The hook order
   is correct by inspection (`v_proj` capture occurs during the current
   forward), but the test would guard the key promise that future layers never
   reuse baseline V.

## Confirmed behavior

- Paired R implements the specified per-head/query transfer: it removes
  `(1-lambda) A_H`, redistributes that mass according to the destination's
  current proportions, leaves every other key unchanged, preserves total mass,
  and rejects a nonzero transfer into zero destination mass
  (`causal_contrast.py:173-181`). The selected/destination disjoint-mask check
  is present.
- GQA uses Llama's contiguous KV-head grouping. The revised reshape/einsum
  (`causal_contrast.py:182-184`) avoids an expanded `[K,H,D]` value tensor. An
  independent float64 comparison against the former expanded reference matched
  route/content results for 4/2, 8/2, and 6/3 attention-head/KV-head layouts.
- `native_forward` captures current V at `v_proj`, so later-layer values are
  recomputed after earlier gates. It clones and reprojects the full O input,
  preserving the existing full-shape sham pattern. One attention plus one MLP
  gate per layer permits the specified joint replay; MLP output scaling leaves
  the residual branch and downstream recomputation intact.
- Hook ownership is sound: all locally registered handles are removed in
  `finally`. The supplied exception-path test and an independent malformed
  donor run left no forward or pre-forward hooks behind.
- The scope and architecture are appropriate: the backend has no population
  coupling and does not claim to be a semantic reader or complete detector.
  No security finding applies to this in-process, locally validated API.

## Verification

Ran on CPU using the requested interpreter and four OpenMP/OpenBLAS/MKL
threads:

```text
python -m pytest -q tests/test_causal_contrast.py tests/test_native_audit.py
15 passed in 21.69s

/tmp/research_lint_20260912/bin/ruff check route_graph/causal_contrast.py route_graph/native_audit.py tests/test_causal_contrast.py tests/test_native_audit.py
All checks passed!
```

Additional manual regressions confirmed that a future key, a cross-length
same-shaped donor, and an integer donor are all currently accepted; owned hooks
were clean after the runs. No GPU work or code changes were made in this review.

## Re-review after donor and causal-domain revisions

The preceding findings remain the historical record. This section evaluates
the revised `CapturedValues` contract and the corrected mixed-visibility rule.

**Result: Critical 0, Required 1.** The former non-floating/type, wrong
input-ID/shape, model, layer/key-order, and all-future-no-op findings are
resolved. One value-provenance gap remains under the stated “exact captured V”
contract.

### Required

1. **`CapturedValues` metadata does not prove that its mutable tensor is the
   captured V** — `route_graph/native_audit.py:16-25, 111-130`.
   The frozen dataclass exposes a mutable `values` tensor and can be recreated
   with `dataclasses.replace`. A replacement such as
   `replace(capture, values=torch.zeros_like(capture.values))` retains matching
   IDs, layer, ordered keys, model identity, dtype, shape, and finiteness, so
   the recipient accepts it; I reproduced a non-sham recipient replay. This
   can again create a fabricated donor-mediated effect while carrying all
   checked metadata. If the backend promises an *exact captured V*, retain an
   opaque/registered capture token with a backend-owned value digest (or an
   equivalent immutable capture registry) and reject a tensor that is not the
   registered capture. Add a same-metadata substituted-value and an in-place
   mutation regression test. If callers are intentionally trusted to construct
   arbitrary measurement tensors, explicitly narrow the public claim to
   caller-supplied branch metadata rather than exact captured provenance.

### Optional

1. **Specify whether `input_scale` is audit-only or an enforced donor
   condition** — `native_audit.py:17-25, 141-150, 158-168`. It is correctly
   captured and a donor scale normally differs from the unscaled recipient,
   so equality with the recipient would be wrong. It is not otherwise checked:
   a same-metadata capture whose `input_scale` was reconstructed as an invalid
   value is accepted. If eta/branch identity is an enforced contract, add an
   expected donor-scale/branch field to `NativeGate` (or the call) and validate
   it. If it is reporting metadata, document that explicitly.

### Resolved findings

- `CapturedValues` now checks exact input IDs (and therefore length), layer,
  ordered key tuple, same model identity, floating dtype equal to the model,
  declared GQA value shape, and finite values before any hooks are registered
  (`native_audit.py:111-130`). This resolves wrong branch/model/key ordering
  and integer/non-tensor donor acceptance.
- The causal-domain implementation now has the correct rectangular semantics.
  It rejects only a selected set wholly future to every query
  (`:101-102`), permits a history key that is invisible to an early query but
  visible to a later query, and verifies every actual future `(query,key)`
  attention weight is exactly zero (`:198-204`). Independent runs confirmed a
  mixed `(queries=(2,4), selected=(3,))` gate leaves logits through position 3
  unchanged and affects the later causal position; all-future content and
  route gates now raise `ValueError`.
- The existing full-shape O projection, GQA mapping, dynamic current V capture,
  MLP joint replay, dtype checks, and `finally` hook cleanup remain correct.

### Re-review verification

```text
python -m pytest -q tests/test_causal_contrast.py tests/test_native_audit.py
17 passed in 25.88s

/tmp/research_lint_20260912/bin/ruff check route_graph/causal_contrast.py route_graph/native_audit.py tests/test_causal_contrast.py tests/test_native_audit.py
All checks passed!
```

The manual mixed-domain, all-future rejection, substituted-donor, and malformed
scale runs left no owned hooks behind. All verification used the requested CPU
interpreter and four OpenMP/OpenBLAS/MKL threads; no GPU work or code changes
were made.

## Final re-review: captured-value integrity and scale identity

This review applies the clarified boundary: the API detects ordinary research
caller mistakes and cached-value corruption. It is not a security boundary
against a caller that deliberately forges both a tensor and its hash, and it
does not require hashing all model weights for every forward.

**Result: Critical 0, Required 0.**

- Every captured V now records a SHA-256 digest of its contiguous raw bytes
  (`native_audit.py:44-46, 175-187`). Recipient validation recomputes that
  digest after the existing exact ID/layer/key/model/dtype/shape/finite checks
  (`:127-148`). This rejects both an in-place value mutation and a replacement
  tensor with otherwise matching metadata, resolving the remaining
  captured-value integrity finding within the stated trusted-caller scope.
  Viewing values as `uint8` before conversion to NumPy also works for fp16,
  bf16, and fp32 captures without changing their representation.
- `NativeGate.expected_input_scale` now makes the donor's recorded scale an
  explicit recipient condition (`:30-41, 123-148`). Non-donor gates reject the
  field, while a donor must match it exactly. This resolves the prior ambiguity
  between audit-only and enforced eta/branch metadata.
- The donor tests now cover both same-metadata zero replacement and in-place
  mutation, as well as a mismatched scale (`tests/test_native_audit.py:119-138`).
  The prior mixed-visibility and hook-cleanup behavior remains intact.

### Final verification

```text
python -m pytest -q tests/test_causal_contrast.py tests/test_native_audit.py
17 passed in 24.31s

/tmp/research_lint_20260912/bin/ruff check route_graph/causal_contrast.py route_graph/native_audit.py tests/test_causal_contrast.py tests/test_native_audit.py
All checks passed!
```

No GPU work or code changes were made during this final re-review.
