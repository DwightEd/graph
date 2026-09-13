# Source-event graph CPU compiler review — 2026-09-13

Reviewed only `route_graph/source_event_graph.py`, its stated design, and
`tests/test_source_event_graph.py`. This is reference-integrity engineering
review of a CPU-only prototype. It is not a same-event, E/C/N, reader, or native
mechanism assessment.

## Result

**Critical 0, Required 3.**

### Required

1. **Raw pointer predictions have no input-inventory identity, so a prediction
   from one source can be compiled as citations into another source.**
   `compile_pointer_events(inventory, prediction)` accepts exactly
   `{events}` (`source_event_graph.py:68-71`). The prediction carries neither
   source/sample identity nor the raw inventory hash. If two sources have a
   compatible unit/token shape, a caller can accidentally route source A's
   pointer output to source B and receive a fully `reference_integrity`
   graph containing B's quotes. The output is tied to B's inventory afterward,
   so recompiling it cannot reveal the original mix-up.

   Require an expected inventory SHA-256 (and, preferably, side/sample ID) in
   the prediction envelope and compare it before resolving pointers. Include
   that binding in the graph/recompile validation. A source/response graph
   must remain independently frozen even when their token geometry matches.

2. **Role pointers are non-overlapping but their supplied order is not
   validated.** At `source_event_graph.py:83-88`, every span is checked for
   overlap, but an event with a predicate pointer before its earlier subject
   pointer is retained. The approved interface requires role pointers to
   preserve raw order as well as stay in the declared unit and avoid overlap.
   Either reject decreasing role spans or canonicalize only after proving the
   supplied order; do not silently turn a malformed sequence into a valid
   source event. Add a regression with reversed subject/predicate pointers.

3. **An invalid top-level prediction is discarded by exception rather than
   retained as a compilation failure.** A non-`{events}` root raises before a
   graph is built (`source_event_graph.py:68-72`). Individual malformed event
   proposals are correctly preserved in `failures`, but malformed model roots
   are common boundary failures and the design requires failures and uncovered
   units to stay in the source-graph manifest. Return an immutable graph with
   the raw serializable prediction, a root-level failure record, zero retained
   events, and raw inventory coverage; reserve exceptions for invalid compiler
   arguments or non-serializable input.

## Confirmed behavior

- The raw inventory covers the complete Python-character text, records every
  lexical/punctuation token with exact spans, and labels this as raw-character
  coverage rather than fact completeness. Empty/uncovered inventory content
  cannot authorize absence inference.
- Retained pointer quotes, node IDs, event IDs, gaps, and provenance come from
  the supplied raw inventory; no reader, free quote, natural label, or native
  score is used. Parsed two-role intransitive events and partial events are
  distinguished without inventing a missing object.
- Literal source fields are parsed with `ast.parse(..., mode="eval")` and a
  narrow recursive whitelist. Calls, names, attributes, operators,
  comprehensions, unpacking, tuples, duplicate keys, non-finite values, and
  `-True` are rejected; nothing is evaluated. The conversion from AST UTF-8
  byte columns to Python character offsets correctly preserves Unicode
  citations in the exercised fixture.
- `None` is retained as an observed literal with epistemic status `unknown`;
  it is not converted into a missing-source or false claim. Individual bad
  event proposals are retained as failures and raw/event coverage remains
  separate.
- Inventory and graph digests are immutable JSON hashes, and validation
  recompiles from the raw inventory rather than trusting the graph's digest.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_source_event_graph.py
19 passed in 0.14s

/tmp/research_lint_20260912/bin/ruff check route_graph/source_event_graph.py \\
  tests/test_source_event_graph.py
All checks passed!
```

I also independently confirmed that reversed roles are retained, a
same-shaped foreign-source prediction compiles, and a malformed root raises
`ValueError`. No GPU work or frozen-v3 route changes were made.

## Required re-review — 2026-09-13

**Critical 0, Required 0.** The three Required findings are closed.

`compile_pointer_events()` now accepts only an inventory-bound request envelope
containing the frozen inventory SHA-256, side, sample ID, and the model's
events-only prediction. It validates those identity fields before parsing and
retains the envelope in the immutable graph, so an otherwise same-shaped
source/response inventory cannot reuse another graph's pointer output.
Role spans must appear in increasing raw order. A malformed model root now
produces a zero-event failure graph with the original envelope and full raw
coverage denominator; invalid compiler arguments and identity mismatches still
raise. `validate_pointer_graph()` recompiles through the bound envelope.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_source_event_graph.py
26 passed in 0.09s

/tmp/research_lint_20260912/bin/ruff check route_graph/source_event_graph.py \\
  tests/test_source_event_graph.py
All checks passed!
```
