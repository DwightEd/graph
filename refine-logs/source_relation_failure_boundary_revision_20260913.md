# SourceRel failure refinement — implementable boundary revision

2026-09-13. Targeted `research-refine` revision after the actual M2 failures and the follow-up objection that the prior late-interaction proposal still contained undefined query-side semantic providers. This is a method boundary note, not an implementation change. No GPU, labels, downloads, or frozen-code edits.

## Correction to the previous proposal

The previous proposal correctly identified the structural failure of single-vector SourceRel-Mini and the need for complete source inventory, component provenance, unknown-valued owners, and bundles. It was still too strong in three places:

1. `q_relation_tokens`, `q_owner_tokens`, `q_condition_tokens`, and `missing_required_condition` were not defined without an oracle semantic parser. They cannot be used as hard scoring channels or failure gates.
2. `masked_parent_all_surface_slots` is unsafe because the current surface slots include names, function-like content runs, and words such as “offering”. Masking all surface slots can delete the exact owner/context needed to identify a source relation.
3. A source bundle containing seven day keys is not proof that the response requires all seven days. Query-side quantifier/condition requirements only exist when a typed verifier proves them from the response text.

Therefore the next implementable architecture must be narrower.

## Revised minimal architecture

### Stage 0: source inventory first, no scorer yet

Implement complete source inventory before another trained or GPU ranking run:

- Preserve every source field/occurrence as a graph node, including long text parent nodes and unknown-valued attributes.
- Split exact components only when the program can preserve provenance to the original field/span. A component is `surface_component`, not a semantic role.
- Keep bundles for field record, review tuple, address/location, attribute key, weekly schedule, and source sentence/clause. Unknown payload does not delete the owner; it only blocks value-level S/C/N.
- Record omitted/unsupported nodes in the denominator by source ID and field path.

This stage directly fixes the observed impossibility: no masked-query model can retrieve a review-text or unknown-WiFi owner if that owner was excluded from the candidate graph.

### Stage 1: query view with only target-coordinate masking

The response-side query should not require SRL or relation buckets:

```json
{
  "response_id": "...",
  "parent_text": "complete sentence/readable base unit",
  "prior_context": "all previous response text",
  "target_slot_id": "...",
  "target_span": [start, end],
  "target_surface_type": "number|time|date|content_run|explicit_quote|unknown",
  "target_masked_context": "same raw context with only this exact target span replaced by <VALUE>",
  "heuristic_hint_spans": [
    {"kind": "lexical_proposal", "span": [a,b], "label": "rating|location|hours|unknown"}
  ],
  "target_value_visible_to_owner_score": false
}
```

Only the exact target coordinate is masked. Prior context and non-target words stay visible. Heuristic hint spans may be produced by rules, but they are proposals and cannot serve as oracle labels, hard inclusion criteria, or gold role buckets.

### Stage 2: source-side channel views, not query-side oracle buckets

Channels can be defined on the source side because the source graph supplies field paths and parent structure. The query remains one raw token sequence.

For each source bundle/member, build independently encoded views:

```text
source_owner_view      = record/business/review parent context, target payload masked
source_relation_view   = field path aliases, attribute key, bundle type, units
source_condition_view  = day/stage/reviewer/negation/quantifier fields present in source
source_payload_view    = original value and exact source span; excluded from owner score
```

The owner score uses token-level interaction from the raw masked query tokens to these source-side views:

```text
S(q, b, m) = α_o * LI(q_raw, owner_view(b))
           + α_r * LI(q_raw, relation_view(b,m))
           + α_c * LI(q_raw, condition_view(b,m))
           + α_t * type_compat(q.surface_type, m.value_type)
           - penalty(identity_only_match)
```

`LI` may be ColBERT-style late interaction: encode query/source separately, compute token-level similarities, reduce by a fixed MaxSim/top-k pooling rule. This is a retrieval feature, not role truth. The score components mean “raw query tokens matched a source-side view,” not “the query has a proven owner/relation/condition token.”

`identity_only_match` is computable without an oracle: if owner-view score is high while relation/condition/source-type evidence is low, return `ambiguous_identity_match` rather than a confident owner. The exact threshold must be frozen before evaluation and reported as a heuristic, not calibrated truth.

### Stage 3: bundle beam with proposal semantics only

For multiple slots in one parent, do not force all slots to the same event. Use a soft compatibility factor only when the source graph proves candidate bundles are compatible and the response surface gives a weak reason to group them.

```text
E(z) = Σ_slots -S(q_slot, z_slot)
     + β * fixed_source_incompatibility(z)
```

No one-to-one source capacity. No claim that MaxSim proves bidirectional provenance. The output contract must say:

```json
{
  "slot_id": "...",
  "owner_candidates": [...],
  "bundle_candidates": [...],
  "score_components": {"owner_view": 0.0, "relation_view": 0.0, "condition_view": 0.0, "type": 0.0},
  "status": "candidate_owner_only|ambiguous_identity_match|missing_source_inventory|unknown",
  "semantic_truth": null,
  "native_certificate_count": 0
}
```

### Stage 4: typed/finite verification is the only source of condition requirements

A source weekly-hours bundle may contain all seven day keys, but that alone does not prove the response claims all seven days. `required_conditions = 7` is valid only if a typed quantifier/condition parser verifies “every day”, “daily”, or an equivalent pattern in the response. If no typed verifier fires, the bundle remains a candidate owner and condition scope is `unverified`.

Similarly, unknown-valued attributes such as WiFi can be applicable owner candidates with factual-status unknown. They should not be dropped at candidate time; they should be blocked only from value-level S/C/N or exact B/A construction until a verifier can compare the payload.

## Evaluation boundary

Do not use LLM self-judged owner labels as gold. If human owner annotations are later available, they can evaluate candidate recall, but their absence should not become a hard gate that blocks all progress.

For the next iteration, report predeclared denominators instead of arbitrary pass/fail thresholds:

- total source fields and bundles; omitted long text/unknown/component counts;
- total response slots; candidate pool available/unavailable by task/type;
- ambiguous identity-only matches;
- finite verifier applicable/support/conflict/unknown counts;
- verified B/A yield with reasons for failure;
- source-seen versus source-heldout status;
- fixed failure-family outcomes for 6303, 6301, 9271, and all-days hours.

No “10 B/A on 36” blanket stop rule. Whether to spend native GPU should be a resource decision from these denominators and failure reasons, not a hidden success criterion.

## SourceRel-Mini validation status

The `0.9539877301` source-validation top-1 is exploratory evidence for source-field reconstruction, not proof of natural ownership. The old validation was already used during design decisions and cannot by itself justify another ranker run or a method claim. Future source-side validation should be reported for diagnostics, but the decisive question is whether the natural candidate inventory and finite verifier yield actually improve under frozen, label-independent conditions.

## Implementation order recommendation

1. Implement complete source inventory/components/unknown/bundles only.
2. Run CPU inventory diagnostics on the old 36 and full population: show previously missing review text, unknown attributes, address/location components, and schedule bundles are present or explicitly unresolved.
3. Freeze the target-coordinate masked raw query view. Do not add all-slot masking.
4. Only after query scoring is defined as raw-query-to-source-view late interaction with fixed source-side channels should a scorer be coded or encoded on GPU.
5. Keep Qwen/finite readers out of owner scoring; use them later only for bounded relation/value verification among frozen candidate IDs.

This preserves the accepted insight from the previous report while removing the undefined automatic semantic providers that would otherwise recreate the same sourceQA/SRL bottleneck under a new name.
