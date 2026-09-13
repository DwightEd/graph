# Atomic-anchor v2 integration review — 2026-09-13

Reviewed `route_graph/atomic_anchor.py`, its integration through
`audit_prepare.py`, protocol v2, runner code freezing, and
`tests/test_atomic_anchor.py`. This was CPU-only and did not modify the
implementation, frozen population, or output artifacts.

## Result

**Critical 0, Required 1.**

### Critical

None.

### Required

1. **The advertised atomic budget is not the enforced extraction-call
   budget** — `route_graph/atomic_anchor.py:237-250, 278-289, 361-362` and
   `route_graph/audit_protocol.py:5-11`. Protocol v2 freezes
   `atomic_units_per_batch`, `atomic_frames_per_batch`, role/predicate limits,
   and reader JSON limit, but the compiler uses independent literals (`4`,
   `24`, `16`, `6`, `1536`). More importantly, `question_limit=48` truncates
   only *after* every response unit has already issued a FRAME request with up
   to 24 proposed frames. A long response therefore makes an unbounded number
   of extraction calls and raw frame candidates before it reaches the frozen
   question cap. Pass the protocol limits into the compiler and define/enforce
   a response-level extraction-call/frame budget, recording an explicit
   `atomic_budget_unresolved`/coverage state for units not processed. This is
   needed for the stated bounded A-stage request contract; silently retaining
   only the first 48 compiled questions does not bound reader work.

## Confirmed integration behavior

- The compiler builds JSON questions with exactly one requested role omitted;
  it checks direct target and resolved-referent aliases before serializing
  conditions/premises. `_candidate_question()` then verifies the exact response
  spans and rejects leaked values again.
- A referent must be an exact earlier response span. The decontextualized value
  is retained as a condition/premise for other targets, while a target whose
  alias would be exposed is rejected individually rather than broadening the
  event.
- `AtomicReader` replaces only extraction/self/source/verify prompts. Source
  and blind verification still receive source, task, and answer-hidden
  question; source answers are not passed between them.
- Condition rows must provide exactly one exact-value row for every non-target
  role. Invalid, duplicate, extra, or unsupported rows retain the raw
  prediction but force an uncertain effective status. In particular, a
  `not_stated` target with a missing non-target premise cannot become N.
- The compiled question retains the legacy fields used by `build_anchors`, so
  exact answer spans, edits, full events, token slot masks, and evidence
  anchors flow through `prepare_anchors`. It additionally carries atomic role
  bindings. The existing C/D prior-question mechanism still consumes specific
  question IDs and `answer_span`s, so multiple atomic targets of one claim can
  be slot-linked without treating the whole clause as the origin.
- `atomic_anchor.py` is in the runner code manifest; protocol v2 changes the
  settings digest, preventing reuse of v1 A–D artifacts.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_atomic_anchor.py tests/test_evidence_anchor_review.py \\
  tests/test_audit_pipeline_review.py tests/test_audit_alignment_review.py
19 passed in 17.15s

/tmp/research_lint_20260912/bin/ruff check route_graph/atomic_anchor.py \\
  route_graph/audit_prepare.py route_graph/audit_protocol.py \\
  route_graph/audit_runner.py tests/test_atomic_anchor.py
All checks passed!
```

## Final re-review: enforced atomic budget — 2026-09-13

**Result: Critical 0, Required 0.**

The atomic budget finding is resolved. `prepare_anchors()` passes the frozen
`PROTOCOL` explicitly. `extract_atomic_questions()` consumes its batch-unit,
per-batch frame, per-response frame, per-response batch, JSON-token,
role-word, and predicate-word limits; `AtomicReader` likewise obtains the
source JSON limit from the same protocol. Invalid frames consume the
per-response frame cap, so a malformed frame cannot create extra reader work.

`atomic_budget` records executed batches, considered frames, returned frames
omitted by the cap, and every unprocessed unit ID. It reports `exhausted` when
either unit or returned-frame work was truncated. Such units are never added to
the nonassertion list, so normal anchor coverage leaves them explicit as
`coverage_unknown`.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_atomic_anchor.py tests/test_audit_pipeline_review.py
17 passed in 15.60s

/tmp/research_lint_20260912/bin/ruff check [atomic anchor, prepare, protocol, tests]
All checks passed!
```

No GPU or implementation changes were made in this re-review.
