# Typed hours owner metadata correction review — 2026-09-13

Scope: bounded impact review of the discovered `root_owner_id` metadata bug in frozen typed-hours outputs. I did not edit frozen code or results, did not run GPU, and did not add experiments.

## Finding

`next_iteration/typed_hours.py::source_schedule()` writes:

```python
"root_owner_id": graph["literal_graph"]["nodes"][0]["id"]
```

For the current literal graph, `nodes[0]` is the scalar `name` node, not the root dict. The actual record root is `literal_graph["root"]`, whose node has empty `path: []`.

Confirmed in frozen `typed_hours_population_prepare_20260913` for both compiled native inputs:

| response_id | source_id | stored `root_owner_id` | actual `literal_graph.root` | `nodes[0].path` |
|---|---|---|---|---|
| 9273 | 14637 | `20a28f57...` | `642b9b77...` | `['name']` |
| 8420 | 14289 | `bfb12fd3...` | `3f8e8623...` | `['name']` |

The same bad ID is copied into `fact["constraint_graph"]["owner"]` for the typed contrast facts.

## Impact assessment

This is a metadata identity bug, not a typed-value or native-input bug.

Unaffected:

- Source schedule parsing of seven day intervals.
- Endpoint minutes and `end > start` checks.
- `source_value_spans` for the edited endpoint.
- Seven required day occurrence IDs.
- Exact response B/A text construction.
- Tokenizer exact-prefix validation.
- Source endpoint token keys used by native search.
- Native forward counts, if run, because the compiled contrast uses `source_keys`, not `constraint_graph.owner`.
- WiFi support check, which uses `schedule["wifi"]` and `wifi_occurrence`, not `root_owner_id`.

Affected:

- Human/machine interpretation of `constraint_graph.owner` as the record root. It currently points to the `name` scalar anchor.
- Any downstream grouping, audit table, or written claim that says the owner ID is the root record node.
- If later code uses `constraint_graph.owner` as a graph node for root-level record grouping, it will group by name scalar rather than record root.

Semantic validity is not changed for the current typed-hours facts. The response grammar accepts either the explicit business name or unqualified `The business` under the declared Data2txt single-record assumption. Using the `name` scalar as an anchor is still evidence for the business identity, but it is misnamed as the record root and should not be presented as a root-record owner node.

## Correction plan without rerun

A full 17,790-row CPU prepare rerun or GPU native rerun is not necessary because the mathematical proof objects and native inputs are unchanged. Use an additive correction manifest that leaves every frozen artifact hash intact.

Recommended manifest file:

```text
outputs/typed_hours_population_prepare_20260913/metadata_corrections/owner_root_20260913.json
```

Recommended fields:

```json
{
  "schema": "typed-hours-owner-root-correction@1",
  "reason": "source_schedule.root_owner_id used literal_graph.nodes[0].id, which is the scalar name node; literal_graph.root is the dict root",
  "frozen_run": "typed_hours_population_prepare_20260913",
  "no_rerun_required": true,
  "frozen_artifacts_modified": false,
  "corrections": [
    {
      "response_id": "9273",
      "source_id": "14637",
      "schedule_path": "sources/...json",
      "fact_index": 4,
      "old_owner_id": "20a28f57...",
      "old_owner_path": ["name"],
      "old_owner_kind": "scalar",
      "old_owner_interpretation": "business_name_anchor",
      "correct_record_root_id": "642b9b77...",
      "correct_record_root_path": [],
      "correct_record_root_kind": "dict",
      "semantic_value_proof_changed": false,
      "native_contrast_changed": false
    }
  ]
}
```

The manifest should be referenced from the run witness/report rather than replacing the sealed schedule, row, or contrast JSON. If a normalized view helper is added, it should return both IDs:

```json
{
  "business_name_anchor_id": "old node0 id",
  "record_root_id": "literal_graph.root",
  "owner_scope": "Data2txt single-record root business",
  "legacy_constraint_graph_owner_field": "business_name_anchor_id in frozen artifacts"
}
```

Future code should write `record_root_id = literal_graph["root"]` and, if useful, separately write `business_name_anchor_id` for the `name` scalar. Do not overload `owner` with both meanings.

## Reporting language

For frozen typed-hours outputs, use this wording:

- Correct: “The typed contrast used seven exact `hours.<day>` endpoint spans and source token keys; the frozen owner metadata field points to the business-name scalar and is corrected by an additive manifest to the dict root.”
- Incorrect: “The frozen `constraint_graph.owner` is the record root.”
- Incorrect: “This changes the 9273 typed proof or B/A contrast.”

## Required status

Critical: 0.

Required before using the frozen results in a paper/report: add the correction manifest or normalized metadata view so no downstream claim treats the scalar name node as the record root.

Required before future reruns: change `source_schedule()` to store `record_root_id = literal_graph["root"]` and optionally `business_name_anchor_id` separately.

No rerun is required for current typed values, spans, token keys, or native B/A inputs.
