# Post-first frozen-population analysis engineering review — 2026-09-13

## Scope

Reviewed `next_iteration/population_postfirst.py` and
`refine-logs/POST_FIRST_ANALYSIS_PLAN_20260913.md`. This is a CPU-only
post-hoc analysis of the completed 17,790-response observer replay, not a new
prediction, training run, GPU forward, detector, or causal result. I added
`tests/test_population_postfirst_review.py` only.

## Required

### R1 — verify against the frozen evaluator, not only its hash

The new analysis hashes every `evaluation_*.json` but never reads an evaluation
or compares its overlapping all-token / through-first-error outputs. Thus a
weighting, masking, field-selection, or score-direction regression could
publish a plausible new analysis without satisfying the plan's required
bootstrap-0 equivalence to the existing source-balanced metrics.

Before annotations are joined, require the uniquely selected completed frozen
evaluation artifact and bind its hash in settings. Before publishing, compare,
for every task × generator × split group and every shared fixed score, the
all-token and through-first-error AUROC/AUPRC to that artifact with a documented
floating-point tolerance. New post-first-only scores may remain additional, but
must not replace the legacy comparison set.

### R2 — fail explicitly unless roster and annotation IDs are equal

The code checks roster cardinality and annotation uniqueness, then indexes
`labels[rid]`. A missing annotation produces an incidental `KeyError`; extra
annotations are accepted silently. The frozen full population calls for a true
one-to-one annotation join.

After loading annotation IDs and before iterating responses, reject unless the
roster and annotation ID sets are exactly equal. Report the bounded mismatch
counts/IDs in the error, without reading labels into a result.

## Confirmed contracts

- `through_first_error` includes the first annotated error; `post_first_error`
  is its strict complement. For no-error responses, all tokens remain in the
  first-inclusive denominator and none enter post-first. The two masks are
  disjoint and partition each response.
- The nine listed scores have fixed higher-is-error signs. No label-conditioned
  direction flip or score normalization occurs.
- The optimized inverse/count weighting equals the old
  `binary_detection_metrics(..., bootstrap=0, source_balanced=True)` calculation
  and avoids unused cluster-bootstrap construction.
- The analysis requires exact COMPLETE/progress success, frozen annotation and
  roster byte hashes, verified response artifacts, source/generator/split/text
  identity, original offsets, finite score shapes, and an end-of-run rehash.
- Outputs freeze code/plan and parent hashes before parsing annotation JSON,
  record every response manifest hash, and use a fresh output directory.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=../reanchor/src:. \
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
  -m pytest -q tests/test_population_postfirst_review.py
# 2 passed in 13.67s

/tmp/research_lint_20260912/bin/ruff check \
  next_iteration/population_postfirst.py \
  tests/test_population_postfirst_review.py
# All checks passed
```

The regressions cover mask partitioning (including no-error responses), every
fixed score sign, and exact source-balanced AUROC/AUPRC equivalence to the old
helper. They are not experiment results.

## Current verdict

**Critical 0 / Required 2**. Do not run the analysis until R1/R2 close. Even
then its output must remain a descriptive observer-replay partition analysis;
it cannot establish ownership, routing, continuous-span detection, or native
causality.

## Targeted re-review closure — 2026-09-13

R1 and R2 are closed.

- The analysis now requires exactly one frozen prior evaluation artifact and
  snapshots its hash. It verifies the prior run-settings hash, completed
  response count, group set and group response denominators. For every one of
  36 groups, both legacy subsets, and all seven shared fixed scores, it checks
  the all/first token and error denominators plus source-balanced AUROC and
  AUPRC at `atol=rtol=1e-12` before publishing. This gives 504 independently
  checked AUROC/AUPRC metric pairs. The two new post-first scores remain
  explicitly additional rather than silently replacing old metrics.
- The annotation and roster IDs must now be exactly the same set after their
  respective uniqueness checks. A missing or extra label fails at the join
  boundary instead of producing an incidental lookup error or being ignored.

The focused CPU tests still pass and Ruff remains clean:

```text
tests/test_population_postfirst_review.py: 2 passed in 15.06s
ruff check next_iteration/population_postfirst.py
  tests/test_population_postfirst_review.py: all checks passed
```

Final targeted verdict: **Critical 0 / Required 0**. Running this code may
answer the bounded post-first descriptive-signal question only; it remains
outside ownership, routing, continuous-error detection, and causal claims.
