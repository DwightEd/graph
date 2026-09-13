# Fact-scope engineering review — 2026-09-13

## Scope

Reviewed only next_iteration/fact_scope.py, next_iteration/fact_scope_mask.py and their two tests. This is a conditional-assumption incidence/CSP and coordinate-mask implementation. It is not an automatic parser, semantic certificate, source-search completeness proof, native integration, or a mechanism result.

Independent CPU verification:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
      -m pytest -q tests/test_fact_scope.py tests/test_fact_scope_mask.py

Result: 21 passed in 6.03s.

## Critical

None.

## Required

### R1 — a multi-fact scope table can bypass the focal-factor boundary

solve_owners accepts a table whenever target_fact is merely a member of scope_fact_ids. A table with scope_fact_ids=[target_fact, other_cooccurring_fact] is therefore usable, and can forbid all but one owner even if its asserted relation actually came from the other factor. The existing test covers a table whose scope is only other, but not this broad-scope bypass.

I reproduced this with a tested one-variable table scoped to both f and other: it returns unique_under_supplied_constraints with owner A. This conflicts with the stated contract that a separate co-occurring fact must not constrain the current binding and that non-target errors must not create a different owner.

Minimal fix: for this single-factor solver, require table.scope_fact_ids to equal [target_fact] after canonical sorting. If a future semantic provider needs a genuinely composite factor, it must first compile a separately sealed composite factor with its own members/target/footprint; it must not smuggle extra fact IDs through this field. Add the corresponding regression.

### R2 — owner_payload admits an unmasked source candidate and has no receipt binding

owner_payload verifies only that response_view contains a target_value mask. Every source view is accepted after a generic seal check, even when its text has no candidate_value or linked_value mask. I reproduced a response view with "8" masked and a source view "Source value: 8" with masks=[]; owner_payload emits the raw source value unchanged.

This makes the module's coordinate-mask guarantee dependent on an undocumented caller convention exactly where the payload becomes available to a relation reader. Further, neither owner_payload nor solve_owners binds a table to the exact response/source view SHA set or an immutable reader receipt. A caller can mark a value-dependent table tested while passing a different view.

Minimal fix: require each owner-comparison source view to carry at least one candidate_value or linked_value mask; require the producer of a tested table to supply and verify the exact owner_payload digest (including ordered response/source view identities), masking-manifest digest and reader receipt/request digest before solve_owners consumes it. The source values can reappear only in a distinct, explicitly named later source-support/B-A request, not an owner table.

This does not ask the mask module to discover every semantic dependency. It enforces that the caller's declared candidate-value masking and receipt are actually the inputs to the table which is permitted to prune assignments.

## Confirmed contracts

- An unlisted relation tuple remains possible; only explicit forbidden rows prune. tested allowed rows are not silently treated as a closed world.
- An unverified table makes the result unverified_constraints_remain and cannot exclude a candidate.
- The visit cap counts attempted assignments; a partial first solution returns search_incomplete rather than unique.
- Empty candidate domains have their own status rather than becoming a contradiction.
- Domains allow multiple variables to reference the same occurrence ID; there is no accidental one-to-one constraint.
- Non-contiguous and overlapping fact membership is retained. incidence_windows emits shared_reference_only with licenses_error_propagation=false and emits no risk score.
- Mask rendering is coordinate based, preserves unrelated homographs, records output/input segments, rejects protected-constraint overlap, and seals the view. These are implementation-integrity properties only.

## Optional

None. Do not add semantic accuracy claims, automatic factor extraction, native call accounting, or GPU tests to this conditional solver review. Once R1/R2 are fixed, the only necessary additions are the focused broad-scope and unmasked/tampered-source receipt regressions.


## Re-review closure — 2026-09-13

The two implementation Required items are closed.

- solve_owners now accepts a table only when scope_fact_ids exactly equals [target_fact]. The added broad-scope regression verifies that [f, other] cannot narrow the focal owner.
- owner_payload now rejects every source view without a candidate_value or linked_value mask. The added regression covers the previously accepted unmasked source value.
- The result explicitly carries provenance_status=supplied_assumptions_not_reader_or_symbolic_certificates as well as semantic_validity=not_established_by_constraint_solver, owner_semantically_certified=false and native_certificate_count=0. This is an honest interface boundary for a pure conditional-assumption solver.

The exact payload/mask-manifest/reader-receipt verification remains **Required at the future semantic-provider → native-eligible handoff**, where a real reader or symbolic provider produces a tested tuple table. It is not a remaining defect in this module: fact_scope does not perform a reader request, cannot validate a reader cache, and must not pretend that a JSON status field is such validation. A handoff wrapper must reject a supplied table unless it verifies those exact identities and must be reviewed before the table can support a B/A or native certificate.

Independent re-run:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
      -m pytest -q tests/test_fact_scope.py tests/test_fact_scope_mask.py

Result: 23 passed in 5.93s.

Current bounded verdict: **Critical 0 / Required 0** for fact_scope.py and fact_scope_mask.py as conditional-assumption/masking utilities. This verdict does not certify factor semantics, source coverage, a semantic provider, B/A construction, native routing, or any scientific effect.

