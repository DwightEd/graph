# Event-local matcher CPU prototype review — 2026-09-13

Reviewed only `route_graph/event_matcher.py`, its matcher specification, and
`tests/test_event_matcher.py`, with the pointer-graph compiler treated as the
already-reviewed C0/R0 dependency. This is candidate-ranking engineering
review. The module is not connected to v3, feature extraction, a reader,
relation verification, contrast construction, or native auditing; no claim of
method effectiveness follows from this review.

## Result

**Critical 0, Required 5.**

### Required

1. **A literal `None` can be emitted as a normal `matched` source link.**
   `match_catalog()` preserves `literal_unknown`, but `propose_event()` still
   creates a `status="matched"` candidate and ranks it like every other
   source scalar (`event_matcher.py:184-209`). A source field such as
   `duration=None` can therefore appear in an assignment as a match instead of
   an explicit `unknown`/`null_literal` candidate. This does not yet produce
   E/C/N, but it corrupts the candidate contract needed by later verification.
   Convert literal-unknown field values to an explicitly non-matchable unknown
   candidate (or preserve a distinct `null_literal` status) and prevent it from
   satisfying role-link or identity accounting.

2. **Data2txt field paths are provenance only in storage, not a matcher
   compatibility feature.** Literal candidates expose `field_path`, but the
   actual `cost_terms` contain only hidden, surface, type, and event-context
   costs (`event_matcher.py:190-207`); no path/key compatibility is calculated
   or recorded. Thus fields with unrelated paths are ranked as if their paths
   carried no information, while downstream cannot audit a promised
   `d_field_path` term. Add a bounded lexical/type/path compatibility term with
   an explicit zero/unknown value when no bridge is available. Keep it a
   provenance feature, never a predicate or same-event decision.

3. **The result has no required wrong-event / wrong-record control
   candidates.** Same-value occurrences are retained in each pool's
   `same_value_candidates`, but `event_assignments` never expose
   `candidate_controls` (`event_matcher.py:273-291`). A downstream comparison
   cannot select or report the required same-value-different-event,
   same-field-unrelated-record, or same-type-shuffled-structure controls.
   Derive deterministic controls from the pre-pruning scored catalog when they
   exist, record whether each was retained or only observed, and avoid claiming
   a control when the source inventory lacks one.

4. **Graph and raw-fallback search coverage lacks a per-pool source-unit
   contract.** The aggregate denominator gives total raw units/tokens and
   catalog-node count, but candidate pools contain neither searched/unsearched
   unit IDs/counts nor `graph_only` versus raw-fallback scope
   (`event_matcher.py:215-236, 281-285`). Raw lexical fallbacks are generated,
   yet consumers cannot tell whether a null/unknown result followed a full raw
   search, graph-only retrieval, or a pruned candidate pool. Add frozen
   searched and unsearched raw-unit denominators plus graph-node/raw-token
   candidate counts and retrieval scope per role. Null and unknown must remain
   candidate states, never evidence of source absence.

5. **Feature provenance is syntactically hashed but not bound from a capture
   input to the raw inventories.** `feature_manifest()` validates encoder
   strings as SHA-256 and binds arrays to ordered catalog IDs, which is good;
   however `source_input_sha256` and `response_input_sha256` are accepted as
   arbitrary 64-hex claims and are never connected to the inventories or a
   frozen capture/mapping artifact (`event_matcher.py:89-121`). Equal-shaped
   arrays from a different rendered input can therefore be given a plausible
   manifest and enter ranking. Require a capture record that binds each input
   hash, catalog/inventory hash, feature-ID ordering, token/span mapping,
   model/tokenizer/layer representation, and array digest. Validate that record
   before constructing the manifest.

## Confirmed behavior

- Source-column capacity is absent and the result explicitly states
  `source_capacity_used=false`; reuse across response events is therefore not
  artificially blocked. Same occurrence links within one event receive a
  documented pairwise penalty rather than being silently treated as identity.
- Same-valued mentions in different source events retain distinct node and
  membership IDs; their competition is recorded without an E/C/N or identity
  verdict. Assignments uniformly state `identity_status=not_verified` and
  `downstream_allowed=relation_verification_only`.
- Literal field records remain `literal_field` provenance nodes rather than
  fabricated natural-language predicates. Raw-token fallback exists when a
  graph is absent, and the catalog labels semantic extraction incomplete.
- Top-k channel pruning and bounded beam search record pruned states. The
  output explicitly says the beam is not a global optimum and costs are not
  probabilities. It does not call a reader, consume labels or native effects,
  or infer source absence.
- Arrays require finite float32 values, exact ordered catalog dimensions, a
  common layer/hidden space, and matching array hashes on proposal. Inventory
  and pointer graphs are revalidated before catalog use.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_event_matcher.py tests/test_source_event_graph.py
36 passed in 0.40s

/tmp/research_lint_20260912/bin/ruff check route_graph/event_matcher.py \\
  tests/test_event_matcher.py
All checks passed!
```

## Non-whitespace pooling re-review — 2026-09-13

**Critical 0, Required 0.** The remaining completeness finding is closed by a
clear, tokenizer-invariant pooling contract: raw whitespace has zero feature
mass, while every non-whitespace character in every catalog span must be
covered or mapping fails. Each membership now stores its raw span length,
non-whitespace character denominator, and pooled character mass, so the
choice is visible and recomputable. The event-span regression records 22 raw
characters, 19 non-whitespace characters, and pooled mass 19 for the intended
space-omitting offset pattern.

This is compatible with the preceding fractional treatment of real byte-
fallback overlaps: a covered non-whitespace character has one unit of mass
split among all tokens that cover it. No new semantic target, GPU capture, or
detection claim is introduced.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_event_matcher.py tests/test_span_feature_capture.py \\
    tests/test_source_event_graph.py
44 passed in 12.56s

/tmp/research_lint_20260912/bin/ruff check route_graph/event_matcher.py \\
  route_graph/span_feature_capture.py tests/test_event_matcher.py \\
  tests/test_span_feature_capture.py
All checks passed!
```

No GPU was started and no frozen v3 file was changed.

## Feature-capture re-review — 2026-09-13

The original five Required items are closed:

- literal `None` is a `null_literal` non-match link;
- finite field-path lexical compatibility is explicit, with 0.5 used only for
  unavailable path information;
- deterministic pre-pruning same-value/other-membership and same-field/other-
  record controls are recorded without calling another record unrelated;
- every role pool records searched/unsearched raw units and graph/raw candidate
  counts; and
- manifests recompute immutable capture records that bind catalogs, inventories,
  contexts, token IDs, memberships, shared encoder identity, and array bytes.

The source capture requires an empty response and exact prompt source span;
source and response captures must share the same prompt/source context. The
tiny randomly initialized Llama check establishes only that post-block hook
pooling equals direct layer-output pooling and removes hooks. It is not model
or mechanism evidence.

**Critical 0, Required 1.**

### Required

1. **Overlapping non-special token offsets are accepted and double-weight raw
   characters.** `span_mapping()` checks only that the next offset starts no
   earlier than the preceding *start* (`span_feature_capture.py:43-52`), rather
   than no earlier than its end. Thus `[0, 4]` followed by `[3, 9]` is accepted.
   Both token states then receive weight for character 3, even though the
   mapping says pooling is overlap-character-weighted. The same flawed mapping
   passes capture-record recomputation, so the hash does not repair it.

   Reject overlapping non-`None` token intervals (while continuing to permit
   `None` special/context positions). The added
   `test_mapping_rejects_overlapping_raw_token_offsets` currently fails and
   supplies the regression boundary.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_event_matcher.py tests/test_span_feature_capture.py \\
    tests/test_source_event_graph.py
40 passed before the added regression; the new focused capture run: 2 passed, 1 failed
```

The only edit in this review is the authorized new CPU regression in
`tests/test_span_feature_capture.py`; no implementation or frozen v3 route was
changed.

## Overlap-policy correction re-review — 2026-09-13

The preceding recommendation to reject all overlapping offsets is withdrawn.
An actual local Llama-3.1 tokenizer byte-fallback pattern for `🧬 🦦 café`
contains legitimate overlapping Unicode offsets. The revised mapping correctly
assigns each covered character fractional ownership (`1 / number_of_covering_
tokens`) and the hook now uses an explicit normalized membership-weight matrix
times each post-block state. This removes the invalid prefix/first-last
assumption. Ordinary, partial-overlap, and recorded byte-fallback patterns all
match direct weighted layer-output pooling in the CPU tests; hook cleanup also
holds. These are engineering checks only, not GPU feature or detection results.

The overlap Required finding is therefore closed. A separate completeness issue
remains: **Critical 0, Required 1.**

### Required

1. **Unmapped whitespace is exempted from coverage even when it belongs to a
   catalog event span.** `span_mapping()` rejects only uncovered
   non-whitespace characters (`span_feature_capture.py:65-66`). For a pointer
   event such as `"A crane lifted crates."`, token offsets that omit every
   space are accepted. The event anchor has an incomplete character-mass
   membership, so its member weights sum to less than its raw span length,
   despite the new per-character ownership contract. The capture record then
   faithfully hashes the incomplete mapping rather than detecting it.

   Require every character in every catalog span, including whitespace, to be
   represented by one or more token offsets, or explicitly redefine pooling
   and its stored denominator as non-whitespace-only. The added
   `test_event_span_rejects_unmapped_whitespace_character_mass` expects the
   current stated full-character contract and fails: 5 passed, 1 failed.

```text
/tmp/research_lint_20260912/bin/ruff check route_graph/span_feature_capture.py \\
  tests/test_span_feature_capture.py
All checks passed!
```
