# Native-anchor diagnostic review — 2026-09-13

Reviewed only the read-only `experiments/diagnose_native_anchors.py` diagnostic
and its new CPU regression. It does not assess the running v3 audit's scientific
outcome, modify a prediction, or enter the frozen code manifest.

## Result

**Critical 0, Required 1.**

### Required

1. **Evidence-invalid condition tables are still counted as non-target
   failures.** The diagnostic accepts a branch whenever
   `atomic_condition_binding.valid` is true
   (`experiments/diagnose_native_anchors.py:42-45`), then counts reader
   `missing`, `conflicting`, or `uncertain` condition rows. In the A artifact,
   however, the evidence layer separately records `condition_table_valid`; it
   is false after citation/sidecar validation fails, and such a branch cannot
   support the semantic classifier. The diagnostic therefore reintroduces a
   rejected table as a claimed non-target failure.

   Require both the structural binding and `condition_table_valid is True` for
   every branch used in this statistic. The added regression constructs two
   different target masks for the same raw claim, each with a structurally
   valid but evidence-invalid missing row; it expects zero collapsed spans and
   words. It currently fails, proving the issue.

## Confirmed behavior

- Each reported A artifact is checked against the frozen settings digest, its
  exact roster row digest, its data digest, and its empty A-stage upstream
  set. The input file is checked against its settings hash before rows are
  read. The added tests cover both input and A-artifact mismatch rejection.
- The statistic considers only uncertain questions, only reader prediction
  branches, and only the three stated non-target statuses. It deduplicates by
  the original `claim_span`; word counts use the union of collapsed spans, so
  overlapping masks cannot inflate either denominator.
- All A words and their `assertion`, `nonassertion`, and `coverage_unknown`
  states remain counted. The output explicitly says the metric is a
  reader-reported interface diagnostic and **not** ground truth of multi-role
  errors; `labels_used` remains false.
- The diagnostic is read-only and does not alter A artifacts or reader cache.

## Verification

I added `tests/test_native_anchor_diagnostic.py`.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_native_anchor_diagnostic.py
2 passed, 1 failed

Failure: test_structurally_bound_but_evidence_invalid_tables_are_not_diagnostic_failures
observed multiple_masks_with_non_target_failure_spans == 1; expected 0

/tmp/research_lint_20260912/bin/ruff check experiments/diagnose_native_anchors.py \\
  tests/test_native_anchor_diagnostic.py
All checks passed!
```

No GPU was started. The only edited file is the authorized new CPU test module.

## Required re-review — 2026-09-13

**Critical 0, Required 0.** The condition-failure gate now requires all four
properties on each reader branch: an atomic binding marked valid, an
evidence-level `condition_table_valid` flag, and no `citation_error` or
`reader_error`. Only then can its reader-reported `missing`, `conflicting`, or
`uncertain` rows contribute to a multi-mask span. The statistics remain
deduplicated by original claim span and retain the full word/unknown
denominator.

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_native_anchor_diagnostic.py
3 passed in 7.78s

/tmp/research_lint_20260912/bin/ruff check experiments/diagnose_native_anchors.py \\
  tests/test_native_anchor_diagnostic.py
All checks passed!
```
