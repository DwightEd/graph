# Cloze-anchor and JSON-framing prototype review — 2026-09-13

Reviewed only the unintegrated prototypes `route_graph/cloze_anchor.py` and
`route_graph/json_framing.py`, their two test modules, and the stated cloze
interface review. This does not assess or modify frozen native-v2 files,
`SELF_QA`, source anchoring, `build_anchors`, or evidence-anchor integration.

## Result

**Critical 0, Required 1.**

### Critical

None.

### Required

1. **Recovered framing does not retain the exact raw suffix or raw coordinate
   of the parsed root** — `route_graph/json_framing.py:26-49`. The parser
   calls `raw.strip()` before `raw_decode()`. Thus tail whitespace is silently
   removed from `discarded_suffix`, and a fenced output reports only a boolean
   while dropping the actual `\n```...` fence suffix/prefix. For example,
   `'{"a":1}]%\n  '` records only `']%'`; a fenced object records an empty
   suffix. `raw_sha256` identifies a cache record but cannot itself preserve
   the recovered suffix or let an artifact audit the root's position in raw
   text. Parse against the raw string with an explicit leading-whitespace
   offset, and persist exact `raw_prefix`, `raw_suffix`, and raw root
   `[start,end]` coordinates (or an equivalent lossless framing record).
   Continue treating a fence as `recovered`, never `strict`.

## Confirmed prototype behavior

- Cloze construction replaces exactly the selected target span while retaining
  all other original clause text. Non-target decontextualized roles replace
  their local surface positions, and target aliases are checked over every
  source-visible field before source payload construction. The source itself
  remains untouched and may naturally contain the answer.
- The sidecar carries mask ID, target role, role bindings, role-keyed condition
  bindings with exact question/response coordinates, decontext bindings,
  hidden aliases, and uncovered alphanumeric fragments. Gaps between extracted
  role spans become explicit `unparsed_context_*` conditions rather than being
  discarded.
- `bind_cloze_condition()` binds a condition by role plus its sidecar position;
  it does not use a global substring lookup. The new reflexive case with the
  same referent value for subject and object proves their coordinates remain
  distinct.
- The framing parser rejects incomplete roots, arrays, duplicate keys,
  non-finite values, prose, and a second JSON value. Its accepted non-content
  tail is restricted to `}`, `]`, `%`, and whitespace; markdown fences are
  counted as recovered.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_cloze_anchor.py tests/test_json_framing.py
18 passed in 8.10s

/tmp/research_lint_20260912/bin/ruff check route_graph/cloze_anchor.py \\
  route_graph/json_framing.py tests/test_cloze_anchor.py tests/test_json_framing.py
All checks passed!
```

No GPU work or implementation changes were made. These remain interface
prototypes and must not be described as a completed v3 anchor integration.

## Framing re-review — 2026-09-13

**Critical 0, Required 0.** The sole Required item above is closed.

`parse_framed_json()` now decodes directly from the original raw string using
an explicit start index.  It persists `raw_root_span`, `raw_prefix`,
`raw_suffix`, the root character count, and `raw_sha256`; consequently
`raw_prefix + raw[raw_root_span[0]:raw_root_span[1]] + raw_suffix` reconstructs
the input byte-for-character at the Python-string level.  A markdown fence and
a permitted terminal punctuation suffix are marked `recovered`. Leading or
trailing whitespace alone may remain `strict`, but is retained in the lossless
record. The new regressions cover whitespace, fenced output, and terminal
punctuation reconstruction.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_cloze_anchor.py tests/test_json_framing.py
21 passed in 8.17s

/tmp/research_lint_20260912/bin/ruff check route_graph/cloze_anchor.py \\
  route_graph/json_framing.py tests/test_cloze_anchor.py tests/test_json_framing.py
All checks passed!
```

This re-review addresses framing only. The code is still an unintegrated
prototype and makes no claim about a v3 end-to-end integration.
