# Constraint inventory engineering review

2026-09-13. Independent bounded `code-review-and-quality` review of `next_iteration/constraint_inventory.py` against `refine-logs/CONSTRAINT_INVENTORY_PLAN_20260913.md`. Scope was the new source-inventory module only. I did not edit implementation, run GPU, read hallucination labels, or run the planned full 17,790-reference prepare.

## Verdict

Critical: 0.

Required: 0.

The module is ready for the planned CPU full-roster source-inventory prepare, subject to the documented scope: it creates raw-source provenance and lexical/component/bundle candidates; it does not create semantic facts, support labels, query requirements, all-days claims, or native certificates.

## Accepted behavior

- Data2txt sources keep all scalar field owners, including `None`/unknown-valued fields and long `review_text` fields. The inventory stage does not filter address numbers or other components by raw scalar type.
- String fields get exact decoded-character to raw-source-span mappings when the Python literal spelling is supported. Escaped quotes, backslashes, Unicode escapes, octal escapes, and raw-string character mapping are handled through explicit coordinate reconstruction checked against `ast.literal_eval`.
- Unsupported literal spellings fail closed: the whole field owner remains available, `component_mapping_status` becomes `unavailable_preserve_whole_field`, and no fake component spans are emitted for that field.
- QA/Summary sources are no longer whole-task domain unavailable. They keep a full `source_document_owner`, sentence/context bundles, word atoms, and surface components with exact raw spans. These remain lexical provenance, not extracted factual events.
- Weekday bundles expose the seven observed source keys with `query_requirement_verified: false` and `semantic_joint_fact: false`. This correctly prevents source-side weekday presence from becoming a response all-days quantifier proof.
- Every emitted edge points to an emitted node in the tested paths, and every `raw_surface` equals the original source text at `raw_span` for fields, contexts, and components.
- `prepare()` uses `route_graph.audit_runner.rows`, preserving the exact input schema, response SHA validation, duplicate response-ID guard, and digit-ID artifact safety. It freezes input/code hashes before writing source artifacts and rechecks them before publishing the manifest.

## Targeted review tests

Added `tests/test_constraint_inventory_review.py` with four CPU tests:

1. `test_decoded_string_map_preserves_escape_raw_spans` checks escaped Unicode, octal, quote, and backslash raw spans.
2. `test_data2txt_inventory_keeps_unknown_long_weekday_and_address_components` checks unknown attribute owners, long review text retention, address numeric components, weekday inventory semantics, node references, and raw-span integrity.
3. `test_natural_source_inventory_keeps_full_document_owner_and_component_refs` checks QA/Summary-style natural-source fallback, full-document owner metadata, component refs, raw spans, and non-semantic bundle flags.
4. `test_unsupported_concatenated_string_keeps_whole_field_without_fake_components` checks fail-closed behavior for an implicit concatenated Python string literal: the field remains, mapping failure is recorded, and no uncertified components are fabricated.

Command run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest tests/test_constraint_inventory_review.py -q
```

Result: `4 passed in 13.36s`.

## Remaining boundaries for consumers

- `contexts`, `components`, and `bundles` are provenance containers. A consumer must not treat containment, same field, same record, or weekday-key inventory as a verified relation, condition requirement, support label, or absence proof.
- `mapping_failures` currently stores `field_id` and reason; field path and raw span are recoverable through `fields`. If later diagnostics need standalone failure rows, adding `field_path` would be a convenience, not a correctness blocker.
- The inventory does not solve source-to-response ownership. It only removes the previous structural impossibility where relevant source owners were missing before scoring.
