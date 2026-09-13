# Typed native runner engineering review — 2026-09-13

## Scope

Reviewed next_iteration/typed_native_runner.py and experiments/interleave_typed_native.py against typed_hours, typed_native_measure and CausalOracle. I did not run GPU, inspect labels, alter implementation or scan the full upstream data.

## Critical

None.

## Required

### R1 — freeze actual control IDs before any A effect search

settings freezes a complete source control census, but run calls search_queries_layers first and then calls match_controls with the group selected from A position effects. Thus the two actual position/origin control IDs are selected after localization has observed A effects.

The matching function does not inspect control intervention deltas, but the selected query/layer scope is itself effect-selected. This is still an adaptive control choice and conflicts with the accepted typed design's no-post-effect control rule.

Minimal repair:

1. after the first exact baseline/profile forward and before search, select deterministic grade-1 control IDs against the all-A/all-query/all-layer root;
2. freeze those IDs, their census entries and their global matching measurements in the artifact;
3. for an effect-selected group, only recheck whether those same IDs satisfy local structural geometry. If they do not, return raw_native_measurements_uncontrolled_or_not_selective. Never select replacement IDs after effects.

This retains raw native results in sparse-control cases and removes no valid input merely for lack of a certificate.

### R2 — make the full-A typed handoff invariant explicit in the runner

settings recomputes prepare_row and native_contrast and compares the entire sealed bridge, which provides strong indirect binding. Before GraphOracle construction, the runner should also explicitly assert:

    bridge.source_keys == sorted(union(bridge.source_keys_per_day))
    every source key is distinct, 0 <= key < contrast.prompt_length
    the passed search key list equals bridge.source_keys exactly

and retain those checks in the result. This closes the generic-measure primitive's R1: no history/response key, searched day subset or altered token roster can receive a typed correct-A path label.

Valid typed A keys are in the prompt, hence visible to every shared-prefix query. Invalid/misbound keys must fail before search; do not convert genuine native geometry/integrity errors into an abstention class.

## Confirmed contracts

- Parent integrity checks require complete, label-free parent manifest/settings/summary; verify parent settings and summary hashes, input hash, live and executed parent code hashes, and selected bridge artifact hashes.
- Every selected bridge is reconstructed from the exact input row through prepare_row and native_contrast under the same observer tokenizer; the full sealed object must match.
- settings snapshots the local executable code and model manifest before publication, verifies snapshots on reuse, and reruns these checks before completion. The output stores input, parent, model, torch/transformers, protocol and code identities.
- Roster selection is a deterministic train-only one-per-source ordering from parent-native inputs; no native effects, preference or labels participate.
- GraphOracle captures raw bf16 hidden states only on the first already-accounted branch-B baseline forward. Feature NPZ bytes, key coordinates, shapes/dtypes and artifact SHA are recorded; hook removal is in finally.
- CausalOracle enforces the 128 actual-forward cap. The search, content/origin shams, donor provenance, branch prefix equality, exact full-shape shams and donors are serialized in result records and donor manifests.
- measure_path preserves raw A deltas without controls, retains baseline A-preferred cases, reports no independent B routing and aggregation unresolved.
- Result, feature, control and donor artifacts are hashed before manifest publication. The interleave wrapper reruns prepare under the inherited phase lock and reuses the previously audited pidfd pause/execute/restore transaction; it does not introduce label evaluation.

## Validation status

The new CPU/tiny-Llama measure tests remain the relevant bounded execution check:

    tests/test_typed_native_measure_review.py: 2 passed in 15.56s

No runner GPU execution or result claim was made.

The remaining status is **Critical 0 / Required 2**. After R1/R2, the runner can close the earlier full-A handoff requirement; it still measures an observer source-A path, not a source-B routing, MLP aggregation, factuality detector, or original-generator mechanism.


## Targeted re-review closure — 2026-09-13

R1 and R2 are closed.

- GraphOracle completes the exact two-branch baseline before initial_controls is computed. The initial root group is the full typed A/all-query/all-layer group; match_controls uses baseline profile/norms only and executes no A/control intervention. The full initial audit and fixed IDs are written before search.
- The selected query/layer group receives fixed_ids. match_controls filters the originally frozen IDs, records backfill_permitted=false, and cannot select a new control after A-effect localization. A locally unmatched fixed ID leaves fewer than two controls and therefore yields the intended raw/non-certificate status.
- settings recomputes the sealed typed bridge and explicitly verifies seven nonempty per-day key groups, exact union equality, recomputed A equality, prompt bounds and source_mask membership. run also rejects any measured or pending search group whose keys differ from the full bridge A roster.
- Valid source-A prompt keys remain visible to all shared-prefix queries. root_group rejects a response/post-prompt key before search.

I added two focused CPU regressions to tests/test_typed_native_measure_review.py:

1. root_group rejects a non-prompt A key and leaves the baseline call count unchanged.
2. A locally matching third control cannot replace a missing initially frozen control ID.

Full bounded test and lint result:

    tests/test_typed_native_measure_review.py: 4 passed in 13.01s
    ruff check next_iteration/typed_native_measure.py
      next_iteration/typed_native_runner.py
      tests/test_typed_native_measure_review.py: all checks passed

Current runner verdict: **Critical 0 / Required 0** for the reviewed typed-native measure/runner handoff. This closure remains limited to engineering integrity. It does not claim a GPU result, general semantic coverage, source-B routing, MLP aggregation, factuality detection, or original-generator causality.

## Serialized bridge re-review closure — 2026-09-13

The first unexecuted v1 prepare correctly failed before model weights, GPU calls,
or a completion snapshot: a loaded JSON bridge represents the
`ContinuationContrast` tuples as lists, so Python container equality with a
freshly recomputed bridge is false. The values' canonical JSON digests and the
sealed bridge hash are equal.

`settings` now validates the loaded bridge seal first and compares
`digest(recomputed)` with `digest(bridge)`. This preserves a full-content check
while accepting the sole JSON storage normalization. The independent explicit
seven-day source-key, prompt-bound, source-mask, and `event_contrast` checks
remain in place, so this is not a weaker semantic handoff check. The failed v1
output is retained as a pre-execution failure; a fresh v2 snapshot is required
for preparation.

The JSON round-trip regression in `tests/test_typed_hours_review.py` verifies
the exact distinction and reconstructs the event through `event_contrast`.
Together with the pair-census fixture, the focused CPU result is 7 passed in
13.22s and Ruff is clean for the changed modules/tests.

Updated verdict: **Critical 0 / Required 0** for the bounded runner and pair
handoff. No native GPU execution or scientific result has been reviewed or
claimed.
