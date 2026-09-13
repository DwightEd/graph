# Typed native measurement engineering review — 2026-09-13

## Scope and validation

Reviewed next_iteration/typed_native_measure.py and the typed-hours/CausalOracle/native gate interfaces. I added tests/test_typed_native_measure_review.py only; no implementation, GPU, labels, or population files were changed.

Independent CPU checks:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
      -m pytest -q tests/test_typed_native_measure_review.py

Result: 2 passed in 15.56s.

Ruff:

    /tmp/research_lint_20260912/bin/ruff check
      next_iteration/typed_native_measure.py
      tests/test_typed_native_measure_review.py

Result: all checks passed.

The added tiny eager-Llama checks establish only engineering behavior: the all-query/all-layer root and a single-layer refinement are retained within the search cap; no-control A trials remain raw/uncontrolled, keep routing and aggregation unresolved, and remove every hook.

## Critical

None.

## Required

### R1 — runner handoff must bind the generic keys to the typed all-days A proof

search_queries_layers accepts a generic oracle and arbitrary keys, while measure_path trusts the selected group's keys as both the source A position and the raw input origin. This is appropriate for a reusable native primitive, but the typed runner must reject any call unless:

- keys equal the frozen typed-native-contrast source_keys, including the complete seven-day endpoint roster, not a searched day subset;
- each key is in the original shared prompt, is less than prompt_length, and agrees with the frozen source alignment/typed proof;
- the oracle contrast, row/prepared/fact hashes, source schedule/protocol identity, model/code identity and slot masks match the same frozen artifact.

This is an integration Required, not a defect requiring this generic function to read a typed artifact. It prevents a caller from passing a history/response or post-hoc key group and receiving the correct-source A label.

For valid typed A keys, every key is in the prompt and every shared-prefix query begins at prompt_length-1. Thus the earlier concern about a source-A query-bisection child being causally invisible does not apply to the typed domain. The runner should fail invalid/misbound input before search; it should not turn native operator integrity errors into a silent class result.

## Confirmed engineering contracts

- The root uses all shared-prefix queries, all layers and the fixed A keys. It is measured and retained before layer/query refinement, preserving joint effects that do not decompose into single layers.
- Layer bisection, two preallocated query seeds and actual non-adjacent query unions retain parents, pending frontier records and unsearched counts. The selected group is a bounded effect-search choice and the output explicitly disclaims unique lookback identification.
- Search actual calls stay under the 64-forward screen allocation. The caps allocate layer/query/union work before later validation, and the 128-forward total leaves capacity for position/origin trials and complements.
- Correct-A direction is delta <= -.5, while full_continuation_preference separately records B_preferred, A_preferred or near_tied. An A-preferred observer is retained and cannot be reported as an error repair.
- source_control_census retains source leaves, source spans and field paths. match_controls records grade, count, attention mass, embedding norm, causal reach and visibility before controls are measured. Only schema_unrelated_to_hours_and_wifi entries may be selected for a strict control pair; schema_related entries cannot silently serve as unrelated controls.
- measure_path runs source-A position and raw-input-to-same-A-V receiver measurements even with no controls. certificate requires two controls, sham, repeat, half and the A-supporting direction, so the no-control path cannot become conditional_typed_correct_source_V_contribution.
- Raw outputs retain the complete B/A event partition, exact sham/donor geometry supplied by CausalOracle, routing=unresolved_no_independent_B_occurrence and aggregation=unresolved. No source B is fabricated from the opposite endpoint or another typed field.
- CausalOracle provides branch-matched donor provenance, ordered V keys, GQA validation, current values, future-mask checks, finite/dtype checks and hook cleanup; the new tiny-Llama test confirms no leaked hooks over the full search plus raw measurement sequence.

## Optional

None. Do not add source-B routing, MLP aggregation, reader semantics, or label evaluation to this typed measurement core. Once the runner enforces R1, the module can produce controlled or explicitly uncontrolled observer-path measurements; neither outcome alone is a general factuality or original-generator mechanism result.

## Pair-census targeted re-review closure — 2026-09-13

The added two-leaf controls are now correctly limited to complete, schema-grade-1,
same-source leaf records. Pair construction rejects both token-key overlap and
half-open raw-character-span overlap. It derives `member_ids`, `spans`, and
`field_paths` from one ID-sorted member sequence, while `keys` is the stable
deduplicated union. Thus an artifact can unambiguously bind each member ID to its
own span/path. Pairs retain the explicit `noncontiguous_control_only_union`
rationale and do not become facts, owners, or sums of member effects.

`match_controls` continues to retain every individual matching check; fixed IDs
are only filtered locally after effect search, never replaced. The pair census
therefore supplies structural candidates only. A lack of a locally matched pair
still produces raw/uncontrolled output.

I added a CPU fixture covering an excluded `hours` leaf, an unknown literal, a
character-overlapping pair whose token keys are artificially disjoint, and an
out-of-order stable-ID pair. The overlap is excluded and the positive pair's
member metadata stays correctly paired. I also added a JSON round-trip regression
for the typed bridge: tuple-valued continuation fields become JSON lists, but the
sealed digest and `event_contrast` canonical reconstruction remain exact. This
catches use of Python object equality at the runner boundary.

Verification:

    tests/test_typed_native_measure_review.py
    tests/test_typed_hours_review.py: 7 passed in 13.22s
    ruff check next_iteration/typed_native_measure.py
      next_iteration/typed_native_runner.py
      tests/test_typed_native_measure_review.py
      tests/test_typed_hours_review.py: all checks passed

Verdict for this pair-census change: **Critical 0 / Required 0**. This is a
pre-effect structural-control contract only; no GPU measurement, mechanism
certificate, or factuality result follows from it.
