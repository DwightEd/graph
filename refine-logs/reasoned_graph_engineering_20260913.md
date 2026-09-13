# reasoned_graph bounded engineering review — 2026-09-13

Scope: `docs/REASONED_GRAPH_METHOD_20260913.md` and `next_iteration/reasoned_graph.py`, plus targeted review tests in `tests/test_reasoned_graph_review.py`. No implementation files, frozen outputs, labels, or GPU execution were touched.

Verdict: **not ready to freeze until the single Required topology mismatch is resolved or explicitly descoped.** I found no Critical bug in parsing, word denominator retention, invalid pointer isolation, or the semantic-prediction/not-GT contract.

## Required

1. **Graph mode does not expose the component/context containment graph claimed by the spec.**

   The method doc describes the source inventory as `field/context/component + record/bundle` containment and says graph mode adds observed source structure while flat mode keeps the same full source and markers. The implementation keeps inline IDs disjoint and lossless, but `source_view()` only creates aliases for `inventory["fields"]` on literal sources, or `inventory["contexts"]` on non-literal sources. It then filters bundle members through those aliases. As a result, field provenance bundles lose their context/component members, and non-literal context bundles lose component members. In the literal fixture I tested, the complete inventory had 6 fields, 4 contexts, 13 components, and 9 bundles, but graph payload exposed only 6 topology nodes and 2 record bundles; zero component IDs were reachable from `source_topology`.

   This matters because the flat-vs-graph contrast no longer tests the complete observed graph promised in the design. For Data2txt it mostly tests record-level grouping; for non-literal sources it can collapse to context markers with little or no component containment. That is a protocol mismatch, not a semantic accuracy issue.

   Minimal fix: keep inline `<s#>...</s#>` markers non-overlapping, but add graph-only topology nodes/edges for every inventory field, context, component, and provenance bundle using raw spans and parent/member IDs. Include a stable mapping from inline alias to inventory ID. If the intended first version is only field/record topology, update the doc/protocol and denominator names to state that component containment is not supplied to the reader.

## Verified by targeted tests

- Raw source rendering is lossless after removing `<s#>` tags, and inline source ID intervals are sorted and disjoint.
- Flat and graph packets share identical `SOURCE`, `RESPONSE`, word count, model-facing instruction, and response word IDs; graph adds topology only.
- One invalid evidence ID does not erase a fact-level prediction, and a bad repair candidate does not erase the word denominator or the valid semantic prediction.
- Overlapping facts are allowed across facts; word-level risk is the arithmetic mean over covering target facts.
- Malformed top-level JSON and wrong top-level types keep the complete response-word denominator with neutral risk `0.5` and record `response_parse` failures.
- Ranges inside one fact reject overlapping or unordered intervals.
- The output consistently marks predictions as `not_ground_truth`, `labels_read=False`, and `native_certificate_count=0`; no semantic correctness is certified by this module.

## Tests added/run

Added: `tests/test_reasoned_graph_review.py`.

Command run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_reasoned_graph_review.py -q
```

Result: **6 passed, 1 failed**. The failing test is `test_graph_topology_exposes_component_containment_from_complete_inventory`, which captures the Required mismatch above.

A first run from the repository parent failed at import collection because existing tests use `next_iteration`/`route_graph` imports relative to the `graph/` working directory. That is consistent with the current test convention and not counted as a module defect here.

---

## R1 closure after scope clarification

Root intentionally scoped `reasoned_graph` as an **external semantic reference**, not the core internal ownership model. The rendered graph topology is now declared as coarse `field/record` or `text-context` provenance; the full component graph remains in inventory for the internal joint model and is not serialized to Qwen in this module. I updated `docs/REASONED_GRAPH_METHOD_20260913.md` to state this explicitly, including that graph-vs-flat here only tests the coarse external semantic reference.

With that scope, the previous Required topology finding is **closed by descoping**, not by adding component topology. The original R1 remains valid as history: the earlier implementation did not match the previous wording that implied full component containment in graph mode. The current wording and `PROTOCOL["rendered_topology_scope"]` now match the implementation.

Additional targeted check: `packet()` now preserves `PROMPT_BEFORE_SOURCE` and `PROMPT_AFTER_SOURCE`, so the QA/Summary question or prompt context around the source is not silently dropped from the model-facing payload.

Final bounded status: **Critical 0, Required 0** for this external semantic-reference module under the declared coarse topology scope. This is engineering readiness only; it does not claim semantic accuracy, internal ownership identification, or mechanism success.

Updated test expectation: `test_graph_topology_honestly_declares_coarse_scope_without_component_pointers` verifies that component counts remain visible through inventory metadata while the rendered Qwen topology intentionally exposes only coarse source IDs/bundles.

Command run from `graph/`:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_reasoned_graph_review.py -q
```

Result: **7 passed in 18.45s**.
