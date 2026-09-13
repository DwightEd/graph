# Restoration v2 feature capture engineering review — 2026-09-13

Scope: `next_iteration/grounded_graph_restoration_features.py`. I left frozen v1 untouched, did not run GPU or inspect labels, and reviewed only the v2 capture/receipt boundary.

## Findings

**Critical: 0.**

**Required: 1. Capture the actual observer dtype and attention backend in each intervention receipt, then validate them on load.**

`PROTOCOL` says `bfloat16` and `sdpa`, but `capture_empty()` records neither `str(model.dtype)` nor `model.config._attn_implementation`; `load_example()` consequently cannot reject an array captured with a different runtime observer configuration but the same model-file manifest. The original feature receipt already binds both fields. Add the two actual runtime fields to the restoration receipt and enforce their exact expected values during receipt verification. This is a provenance requirement, not a claim about model quality.

## Checks that passed

- `intervention()` derives erased positions from frozen positive-overlap `source_offsets`, independently retokenizes the exact prompt, and requires the two overlap sets to agree. It excludes BOS and requires all erased positions to be strictly within the prompt.
- Whole-token boundary treatment is correct: a token crossing either source character boundary is replaced as a unit and its full prompt span is retained in `boundary_crossing_tokens`.
- Replacement writes only prompt positions. The original packet remains unmodified, and the independently recomputed executed-ID receipt fixes the unchanged response-history suffix.
- `capture_empty()` hooks `model.model.norm`, takes only the existing `query_positions == target_positions - 1`, checks pointer identity with the returned final-norm state, removes the hook in `finally`, converts the saved query to finite `float32`, and binds its bytes.
- The artifact verifier re-verifies the complete original feature parent, immutable original packets and entry roster, output arrays, model files, code snapshot/live bytes, forward census, and original-manifest hash. `load_example()` recomputes the intervention before accepting an array and returns original source X together with erased `query_empty` and retained `full_query`.

## Independent CPU regression

Added `tests/test_grounded_graph_restoration_features_review.py`. A random tiny CPU Llama with a deterministic tokenizer whose source token overlaps outside the source span verifies whole-token replacement, untouched response history, genuine erased-input final-norm `t−1` queries, packet immutability, and hook cleanup.

`pytest -q tests/test_grounded_graph_restoration_features_review.py`: **1 passed**. Ruff passes for the module and this test.

An existing broader restoration-pipeline file currently has two unrelated concurrent-test failures (a changed score expectation and an evaluator fixture without the now-required `code_sha256`); they do not affect this feature-capture regression.

## Closure re-review

The Required item is **closed**. `capture_empty()` now records actual `observer_dtype` and `attention_implementation`; `load_example()` requires the frozen `torch.bfloat16`/`sdpa` pair before accepting `query_empty`. The CPU regression now also asserts that capture records the real tiny-model dtype/backend. The focused test remains **1 passed** and Ruff remains clean.

**Final status: Critical 0, Required 0.** This closes only the v2 feature-capture/provenance boundary. It does not constitute a restoration result, a native-route result, or a natural-error detection claim.
