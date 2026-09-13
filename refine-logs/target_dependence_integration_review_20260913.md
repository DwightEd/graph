# Target-dependence and soft-graph native-phase review — 2026-09-13

Reviewed the repaired `target_dependence.py` and the new
`soft_graph_phases.py` native measurement path. This is an engineering review
of observed-target dependence. It does not treat dependence as a factuality,
error-adoption, route-proof, or v3 outcome.

## Result

**Critical 0, Required 2.**

### Required

1. **`soft_graph_phases.py` currently fails the repository Ruff gate.**
   `response_keys` is assigned but unused at line 114. This is a concrete
   merge/integration blocker under the checked lint command. Remove it (or use
   it in a real invariant) and rerun Ruff.

2. **The central per-claim native-phase contract has no direct CPU regression.**
   There is no `tests/test_soft_graph_phases.py`; the 31 passing tests cover
   target, energy, and matcher units but do not execute `_measure_edge()` or
   `native_phase()`. Add a small eager tiny-Llama or bounded test double test
   that proves: one baseline plus at most two source and one history edge never
   exceeds 80 actual calls; each full edge stays at or below 21; controls are
   selected before semantic/native values and are not replaced; missing controls
   remain an explicit no-two-control result; raw-origin, repeat, half, sham,
   query-localization, and route diagnostic fields retain their declared
   target-dependence-only scope. This is needed before a runner consumes the
   artifact, not as evidence of a method result.

## Target-dependence recheck

The three findings in the earlier target review are closed.

- Constructor budgets require a positive integer, and `origin()` reserves two
  available actual forwards before starting its donor/recipient pair.
- Capture validation now precedes artifact publication and checks requested
  layers, ordered keys, full input IDs, scaling record, model identity, dtype,
  GQA shape, finite values, and exact bytes digest.
- Query localization requires an ordered contiguous interval and emits
  causally-invisible split children under `excluded`, while pending budget-
  unsearched intervals and measured parents remain explicit.

`observed_target()` still sums every original target-token log probability,
keeps the original prefix only, excludes future response tokens, records
boundary overlap, and validates prefix/sham equality. The donor path remains
raw embedding scaling → current V capture → receiving-computation donor gate;
the backend's GQA, causal-mask, finite-value, and hook-cleanup checks apply.

## Soft-graph phase assessment

The configured maximum is coherent: baseline consumes one call; a fully
measured edge has position (1), bounded query search (at most 6 actual calls),
content sham (1), raw-origin sham/origin/repeat/half (2 each), two controls
(2 each), and an optional route diagnostic (1): at most 21 calls. With two
source and one history edge, the total remains at most 64 including baseline,
under the fixed 80-call claim budget. Caches can only lower actual calls.

Controls are selected from frozen feature/length/norm pools before reader
relations and native effects; semantic controls need an unrelated-label
threshold, and no replacement search occurs afterward. Route is explicitly a
within-role diagnostic. Results label every native value
`target_dependence_only` / `raw_origin_mediated_target_dependence`; merge
publishes zero strict A/B certificates and says the target-only batch did not
invoke that separate auditor.

## Verification

```text
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
  /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \\
  -m pytest -q tests/test_target_dependence.py tests/test_soft_graph_energy.py \\
    tests/test_event_matcher.py
31 passed in 14.62s

/tmp/research_lint_20260912/bin/ruff check route_graph/target_dependence.py \\
  tests/test_target_dependence.py
All checks passed!

/tmp/research_lint_20260912/bin/ruff check route_graph/soft_graph_phases.py
F841: unused local variable response_keys
```

No GPU was run and no frozen v3 file was changed.
