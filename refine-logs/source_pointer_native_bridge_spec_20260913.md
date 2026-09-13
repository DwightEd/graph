# source pointer native bridge spec review — 2026-09-13

Scope: bounded method/interface review only. I did not modify frozen code or run GPU. I inspected the real interfaces in `causal_groups.CausalOracle`, `propose_native_groups`, `audit_pools.freeze_pools/select_controls/origin_controls`, `audit_certificates.validate_fixed_group/validate_origin`, and `audit_interactions.paired_role/mlp_interaction`.

## Verdict

The bridge can reuse the old strict native backend with a small orchestration layer. The key is to treat the bridge output as a **single validated contrast object**, not as a new evidence QA pool. Once the bridge has fixed B/A texts and source node keys, all native measurements can use the same F:

`F = log P(original B complete event | shared prefix) - log P(edited A complete event | shared prefix)`.

This directly addresses the missing positive/negative event preference signal that soft target-only dependence lacked. It still does not prove internal independent attribution or abstract pointer binding; attribution is conditioned on finite reader validation and must be evaluated later against independent labels.

## Minimal native flow

### 0. Bridge precondition

Run this strict native path only after the source-pointer bridge emits `contrast_valid=true`:

- original event B is predicted unsupported: `q_original[C] + q_original[N] >= 0.8`;
- edited event A is predicted supported: `q_edited[S] >= 0.8`;
- the **selected frozen source node/record/event** supports A, not just another full-source occurrence;
- non-target roles, owner/conditions, and grammar pass;
- target role is in the allowed v1 set: object, quantity, time, location, duration, attribute, origin, destination;
- subject, bool, quoted/code slot, predicate, negation, condition, whole-event, and multi-role repair are rejected before native.

The bridge provides source node keys, response target role/event span, B/A token branches, answer-slot masks, and metadata. It does not provide ground truth.

### 1. Build the strict contrast oracle

Use the old strict branch oracle, not `target_dependence.TargetOracle`:

```python
contrast = ContinuationContrast.from_sequences([B_ids, A_ids], prompt_length)
oracle = CausalOracle(model, contrast, budget=256, input_limit=4096, artifact_writer=...)
```

`CausalOracle` runs both branches for every measurement and checks shared-prefix/sham identity. This is the correct backend because it scores B-vs-A event preference, while `TargetOracle` only scores the observed target span.

Queries must be `contrast.shared_queries`, the complete shared prefix decision domain. Do not restrict the first run to the answer token or to an arbitrary local window. The existing search may later bisect queries and test non-adjacent unions, but the root group must remain the full shared prefix so long-range influence is not excluded before measurement.

### 2. Native proposal without the old free-QA pool

Call bounded native proposal with the bridge-selected source node keys, not all source evidence candidates from a free-text reader pool:

```python
search = propose_native_groups(
    oracle,
    source_keys=bridge.selected_source_keys,
    screen_calls=64,
)
```

This keeps the source side conditioned on the frozen pointer/literal candidate. The history side remains the full previous-response prefix inside `propose_native_groups`, so forward history competition is still discoverable. The proposal uses actual F effects only; it does not see RAGTruth labels, source-answer strings, or semantic control labels.

Keep all-layer groups as the primary certificate scope. `propose_native_groups` uses all model layers for roots and descendants. Layer bands from `validate_layer_bands` are diagnostics after a position group is certified. A result measured only on three handpicked feature layers or arbitrary bands must not be called automatic layer localization.

### 3. Freeze controls, then C filters only

After native proposal, call the old freezing path:

```python
frozen = freeze_pools(oracle, search, row, alignment, question=bridge.question_metadata)
```

This is where position controls and raw-origin controls are constructed. Then run finite C labels over `frozen["views"]` to mark candidates/controls as applicable or unrelated. C can filter the fixed pools; it cannot search replacements after seeing control effects.

Use the two distinct control families correctly:

1. **Position controls** from `freeze_pools` / `select_controls`: same F, same operation type, matched length, baseline attention mass, and visibility. These feed `validate_fixed_group`.
2. **Raw-origin controls** from `origin_controls`: same origin role, matched input length, embedding norm, and causal reach into the selected V-message target. These feed `validate_origin`.

Do not reuse soft graph v1 controls as origin controls. Soft v1 controls compare each candidate node to its own V path under observed target dependence. `validate_origin` instead asks whether the original input tokens for a source/history origin are mediated through a **fixed selected V-message group** under the B/A contrast. The estimands are different.

If C leaves fewer than two unrelated controls in either family, return the corresponding missing-control status and do not backfill.

### 4. Validate source and history position groups

Select at most one source content finalist and one history content finalist from `frozen["finalists"]`, preserving the measured/ranked order and the fixed controls. Source should overlap the bridge-selected source keys. History must be a previous-response group; if finite history relation is not reuse/derive for the current target, it can still be reported as a position effect but cannot enter pairedR as source-vs-history competition.

For each selected position group:

```python
controls = select_controls(item, semantic_view_labels)
position = validate_fixed_group(oracle, item["group"], [c["group"] for c in controls], item["id"])
```

A source group that supports the edited source-grounded event should have `delta <= -0.5` to count as evidence opposing the original B. A history group that supports the original unsupported event should have `delta >= 0.5` to count as bad-history support. Preserve the opposite sign as data, but do not rename it as the hypothesized mechanism.

### 5. Validate raw-input origin through selected V path

For every position group that passes and has an origin candidate:

```python
controls = origin_controls(origin_keys, position["group"], frozen, semantic_view_labels, origin_role)
origin = validate_origin(oracle, position["group"], origin_keys, controls, position)
```

For source, `origin_keys` should be the bridge-selected source node tokens. For history, `origin_keys` should be the mapped previous answer/role tokens only when the bridge/history finite relation identifies a specific previous role; otherwise leave history origin unresolved and do not upgrade pairedR to exact-origin scope.

Passing origin means: this raw input span has a controlled mediated effect through the selected V-message path. It does not prove an abstract binding pointer. The selected input’s answer payload may itself be what flows through V. Pointer/address identification would need an additional content/address separation test, which is outside this minimal bridge.

### 6. Reuse pairedR for source/history competition

Run `paired_role` only if both sides have the required signs under the same F:

- source position/origin opposes original B: `source.delta <= -0.5`;
- history position/origin supports original B: `history.delta >= 0.5`;
- common query and layer scope exists;
- source destination controls are the same-mass unrelated source groups frozen before C effects.

```python
competition = paired_role(oracle, source_position, history_position, source_origin, history_origin)
```

This tests whether reallocating fixed history attention mass toward the validated source destination has a selective effect relative to matched unrelated destinations. It is a source/history transport witness, not a toy relation demo and not a unique-cause proof.

### 7. Reuse MLP interaction only under the same F

Run `mlp_interaction` only if a source evidence position group opposes B and an MLP group exists from the same native search:

```python
interaction = mlp_interaction(oracle, source_position, frozen["mlp"], source_origin, bridge.slot_masks)
```

Keep the current four-state decomposition and downgrades. If context dominates the slot or the MLP scope is broader than the evidence group, report `broad_MLP_event_interaction`. Do not claim a local aggregation error from a broad branch effect.

## Budget recommendation

Use `budget=256` actual Llama forwards per bridge contrast, with `screen_calls=64` for search. This is tight but enough for one source group, one history group, pairedR, MLP, and layer-band diagnostics in the common case:

| Block | Max actual forwards |
| --- | ---: |
| CausalOracle base B/A | 2 |
| bounded native proposal | 64 |
| source + history `validate_fixed_group` | 24 |
| source + history `validate_origin` | 48 |
| `paired_role` | 12 |
| `mlp_interaction` | 22 |
| source + history layer-band diagnostics | 72 |
| margin / cached variation | 12 |
| Total | 256 |

If budget is exhausted, preserve the exact status: `search_unresolved`, `missing_matched_unrelated_controls`, `missing_matched_origin_controls`, `budget_unresolved`, or `layer_scope_unresolved`. Do not replace missing strict certificates with soft target-dependence results.

For a source-only contrast with no eligible history competition, the same budget has large slack. Keep the budget fixed for run comparability rather than dynamically raising it for “interesting” samples.

## Failure boundaries

The strict native bridge should abstain in these cases:

- bridge contrast invalid or finite verifier uncertain;
- selected source node does not support edited A specifically;
- edited A requires subject/bool/quoted-slot/predicate/condition/whole-event rewrite;
- selected source keys are empty or outside the shared prefix;
- fewer than two fixed position controls survive C filtering;
- fewer than two raw-origin controls survive C filtering;
- source and history groups do not share query/layer scope;
- signs are incompatible with the hypothesized source-opposes-B / history-supports-B story;
- layer-band measurements are not run or fail, in which case only all-layer group scope may be claimed;
- origin mediation passes but no content/address separation evidence exists, in which case source-origin path may be claimed but abstract pointer binding may not.

Every failure stays in the denominator for the strict bridge. It is not evidence that the frozen generator lacks internal source/history information; it only says this minimal contrast certificate could not be established.

## Output contract

For every attempted bridge contrast, save:

- bridge IDs: claim, response event, response role, source node, source membership;
- B/A texts, token branches, common prefix, shared queries, slot masks;
- finite validation distributions and pass/fail statuses;
- native search record: measured groups, invalid groups, unsearched tree/union groups, root order, actual calls;
- frozen position controls and C labels;
- frozen raw-origin controls and C labels;
- source/history position certificates;
- source/history origin certificates;
- pairedR and MLP interaction outputs when attempted;
- `scope="conditional_single_slot_contrast_native_certificate"`;
- `not_ground_truth=true`;
- `abstract_pointer_binding_identified=false` unless a future address/content separation test is added and passes.

## What not to add

Do not add a new network, a learned detector, an answer generator, a second free-QA source pool, synthetic hallucination labels, or a toy “relation affects decision” experiment. The bridge’s value is that it converts an already fixed source pointer into a strict B/A mechanism target. The next empirical question is coverage: how often this bounded strict certificate fires, and whether those certificates align with independent factual labels after all predictions are frozen.

## Addendum: full bridge-source origin pool versus searched subgroups

Future implementation must freeze raw-origin controls for the complete bridge-selected source span before C when the strict bridge wants to claim source-origin mediation for that full source node. The current `freeze_pools` pattern creates source-origin variants only for each native finalist `group["keys"]`. If native search later bisects the bridge source node and the certified position group contains only a subspan, looking up `origin_pool_id(bridge.source_keys, group)` will fail unless that full-span origin pool was explicitly added to `origin_control_pools` and `views` before semantic filtering.

There are only two valid choices:

1. Add a frozen `raw_origin_pool(bridge.source_keys, group, roles["source"], norms, row, alignment)` entry for every relevant source finalist before C, using the same role, norm, reach, view, and no-backfill semantics as existing origin controls; or
2. State that source origin scope is only the certified subgroup `group["keys"]`, and report `source_origin_scope="subspan_origin"` or `full_bridge_origin_unresolved` when the full bridge source span was not frozen.

Do not create origin controls after seeing finite labels or native effects, and do not treat a missing full-source origin pool as a successful full-source mediation certificate.
