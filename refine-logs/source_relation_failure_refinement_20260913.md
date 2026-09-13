# SourceRel-Mini natural transfer failure refinement

2026-09-13. Targeted `research-refine` method review driven by actual SourceRel-Mini failures. I read `docs/CURRENT_METHOD_20260913.md`, `refine-logs/source_relation_encoder_design_refinement_20260913.md`, `docs/SOURCEREL_TRAIN_RUN_20260913.md`, `docs/SOURCEREL_TRANSFER_RUN_20260913.md`, and `refine-logs/source_relation_transfer_doc_witness_20260913.md`. I also checked the cited ColBERT method sections (§3.2-§3.3) from the primary paper. No code edits, GPU, downloads, or label reads.

## Decision

Do **not** continue by tuning the current single-vector SourceRel-Mini natural transfer. The actual result is a clean negative for the architecture, not an implementation or format failure.

The source-only task looked strong overall: validation top-1 `0.9539877301`, recall@5 `1.0`, and NCE `0.0968227681`. But the critical same-field/other-record group is exactly where ownership matters, and there SourceRel-Mini is worse than TF-IDF: trained top-1 `0.7727272727` versus TF-IDF `1.0`, with 55 of 60 total top-1 errors in that group. The natural transfer then fails in the expected ways: 622 Data2txt candidate queries, 997 out-of-task slots, no factual/native claims, and both Data2txt sources are actual training sources. This is not source-heldout generalization.

The failure cases show four architectural breaks:

The read-only diagnostic `outputs/source_relation_transfer_diagnostics_v1_20260913/diagnostic.json` strengthens this conclusion without using labels: it covers all 622 candidate proposals, has `labels_read: false` and `new_model_forwards: 0`, and shows missing source inventory before any ranker can help. Source 13717 has 34 occurrences but only 25 candidates; all three `review_text` fields are omitted by the ≤40-word short-scalar policy, and six unknown attribute occurrences are omitted. Source 14637 omits one 44-word `review_text` and unknown `Music`. The numeric address case has no address-number candidate in the pool, so the same-broad-type numeric pool contains only business/review stars. This proves that no masked-query owner model can retrieve the right owner when the source candidate graph never exposes that occurrence or bundle. Unknown-valued fields such as WiFi can still be applicable owners with factual status unknown; they should not be dropped from relation-owner proposal merely because their payload value is unknown.

1. **Source JSON to natural query shift.** The head learned canonical source-field reconstruction, then received a natural response sentence with a masked surface span. It did not learn that “overall rating” differs from review-star rows or that location phrases should bind city/state/address rather than the business name.
2. **Candidate inventory is too scalar and too short.** Pools such as numeric-only `business_stars` and `review_stars` cannot represent address components, long review text components, or schedule bundles. Adding more short scalars without parent bundles will increase distractors.
3. **Record identity dominates field semantics.** Santa Barbara/California mapping to `name` with ~0.99 mass is not a threshold bug; the representation lets any business-identity cue satisfy the whole query. It has no separate requirement that relation words bind the field family.
4. **Set facts are collapsed to one member.** All-days hours should map to a weekly schedule bundle with seven day constraints and endpoint roles. Ranking Monday as a scalar top item is not enough to construct a valid A/B or ownership explanation.

ColBERT is relevant only as a retrieval architecture clue. Its late interaction encodes query and document independently, then scores each query token by its maximum similarity to document-token embeddings and sums those matches. This preserves fine-grained token matching while allowing document-side precomputation. In this project, that suggests replacing one final hidden vector per view with token-level/bucketed interaction. It does **not** make MaxSim a role-truth oracle, a verifier, or an error detector.

## Minimal next architecture: value-masked bundle late-interaction owner proposer

Replace SourceRel-Mini M2 with a **candidate proposer**, not a truth scorer:

> Given a response parent with exact surface slot(s) masked, rank source owner bundles and member occurrences using value-masked source/query views, token-level late interaction, and fixed source graph compatibility. Then let typed/finite verification decide support/conflict/unknown and let native causal code run only on verified B/A subsets.

### Source inventory

Build source nodes at two levels:

- `occurrence`: typed scalar or component span. This includes number/date/time, explicit quoted strings, address/city/state/name components, and exact short components inside longer text when the source parser can give exact source spans. A component is marked `surface_component`, not a semantic role. Long text fields remain as parent evidence nodes even when they have no componentized payload candidate.
- `bundle`: parent fact group with members, e.g. business identity, location/address bundle, overall business rating, review tuple, attribute pair, weekly hours schedule, unknown-valued attribute owner, or source sentence/clause for unstructured sources. Unknown payload does not remove the owner bundle; it only prevents value-level S/C/N and yields factual-status unknown until another verifier resolves it.

Each node stores four separate views:

```text
owner_view: value masked, keeps field path aliases and parent context
relation_view: field/path/bundle labels, units, condition names, quantifiers
condition_view: day/stage/reviewer/negation/owner constraints
payload_view: original value text and exact source span, used only after owner proposal
```

The owner score must exclude the candidate payload value. Payload can be used by the later comparator to build B/A; it must not pull a wrong generated value toward a source owner.

For Data2txt, the source graph root and record metadata stay fixed. For generic QA/Summary, the bundle can be a whole source sentence/clause plus enumerated surface components. If a relevant event is not extracted into the inventory, the output is `unsearched_or_missing_source_inventory`, not N.

### Response query

Use parser-light surface spans only:

```json
{
  "response_id": "...",
  "parent_text": "complete sentence or readable clause envelope",
  "target_slot_id": "...",
  "masked_parent_target_only": "target span replaced by <TYPE_VALUE>",
  "masked_parent_all_surface_slots": "all enumerated value-like slots replaced by type masks",
  "prior_context": "all previous response text",
  "surface_type": "number|time|date|text|quote|unknown",
  "bundle_scope": "parent sentence/base unit id",
  "target_value_visible_to_owner_score": false
}
```

The target-only view supports local slot assignment. The all-slot-masked view supports bundle assignment without letting other wrong values steer the owner. Do not ask Qwen to output free SRL fields before this step; Qwen can later validate finite relations among fixed candidate IDs.

### Score

Use token matrices, not one vector. With existing local Llama states this can start as frozen token features plus a small 128-d projection if needed; a pretrained retrieval encoder is an option only after this local baseline fails and should not be assumed available.

For response query `q`, source bundle `b`, and member occurrence `m`:

```text
S_bundle(q,b) =
  w_rel   * mean_topk_MaxSim(q_relation_tokens, b.relation_tokens)
+ w_owner * mean_topk_MaxSim(q_owner_tokens,    b.owner_tokens)
+ w_cond  * coverage(q_condition_tokens,        b.condition_tokens)
- penalty(identity_only_match)
- penalty(missing_required_condition)

S_member(q,m|b) = S_bundle(q,b)
                + w_type * type_match(q.surface_type, m.value_type)
                + w_local * MaxSim(q_local_target_context, m.relation/condition tokens)
```

The buckets matter. A high owner/name match cannot by itself rank the business name for every text slot; relation and condition evidence must also be present. Conversely, a field-path match cannot override a different record or incompatible condition. If relation evidence is missing and identity evidence is high, the status is `identity_only_ambiguous`, not a confident owner.

ColBERT-style MaxSim is useful here because response words such as “rating”, “address”, “California”, “Monday”, and “every day” can each find source-side supports without collapsing the sentence into one vector. But the result is still only a retrieval score.

### Bundle beam

For all surface slots in the same parent event, solve a small beam:

```text
E(z) = Σ_r -S_member(q_r, z_r | b_r)
     + β * incompatible_bundle_assignments(z)
     + γ * missing_set_coverage(z, response_parent)
```

There is no source one-to-one capacity. A source bundle can support multiple response slots. The beam returns:

```json
{
  "slot_id": "...",
  "topk": [
    {
      "bundle_id": "hours.weekly_schedule",
      "member_ids": ["mon.close", "tue.close", "..."],
      "score_components": {"relation": 0.0, "owner": 0.0, "condition": 0.0},
      "coverage": {"required_conditions": 7, "matched_conditions": 7},
      "status": "candidate_owner_only"
    }
  ],
  "near_tie": false,
  "unresolved_reason": null
}
```

For a schedule statement such as “every day 8 AM-6 PM”, the candidate should be a weekly-hours bundle with all seven day endpoints. A single Monday member may be returned as a member of the bundle, but not as the whole owner if the response expresses all-days coverage.

## What not to add

- Do not add a deeper GNN or more source-reconstruction losses before natural owner challenge cases improve. The current source task already reaches high validation recall while failing natural ownership.
- Do not enlarge the 32k short-scalar pool without bundle parents, component provenance, long-text parent nodes, unknown-valued owner bundles, and identity/relation/condition bucket scores. More raw scalars will make name/review-star collapse worse.
- Do not use target value text in the owner scorer. It can be compared after owner proposal.
- Do not call source-generated role weak labels natural-owner ground truth. They are training scaffolds or finite checks only.
- Do not promote Qwen finite validation to GT. It may condition contrast construction, but final detection still needs independent evaluation.

## Decisive natural evaluation before native rerun

The next architecture must pass a label-independent owner-proposal gate before any expensive native stage:

1. **Fixed failure regression.** On the known natural failures, top-k must show the intended structural correction:
   - 6303 Santa Barbara/California must not rank `name` as a confident owner; city/state/address bundles must appear with relation evidence.
   - 6301 address-like numeric/text spans must not be mapped to business/review-star numeric pools when no address component exists; missing inventory must be explicit.
   - 9271 overall business rating must rank `business_stars` above review-star rows, or mark near-tie if context lacks the distinction.
   - all-days hours must rank a weekly-hours bundle with seven-day coverage, not only Monday.
2. **Natural owner audit set.** Freeze 100-200 natural Data2txt surface slots sampled before seeing predictions, including the above failure families, and manually annotate only source owner/bundle/member, not hallucination labels. This is evaluation, not training. Report top1/top5 bundle recall, identity-only false positives, same-field-other-record errors, and set-coverage errors.
3. **Source-heldout natural transfer.** Include official test Data2txt sources not present in the source-reconstruction training settings. The old 36 all-source-seen result is useful as a stress test but cannot establish generalization.
4. **Contrast yield.** Count how many natural slots produce verifier-approved B/A with non-target conditions preserved. This must be materially above the old zero/near-zero bridge. If owner recall improves but B/A yield remains negligible, the method has not solved the user’s automatic lookback/contrast bottleneck.
5. **Final detector evaluation.** Only after candidate supply and B/A yield pass should the graph detector be compared with independent RAGTruth labels against old SourceRel-Mini, TF-IDF/frozen-cosine, Qwen finite verifier without graph, and typed-rule bridges. Native effects remain certificates on the verified subset, not the detection score itself.

Suggested stop rule: if the bundle late-interaction proposer fails any two of the four fixed failures, has top5 bundle recall below 0.8 on the frozen owner audit set, or yields fewer than 10 verifier-approved nontrivial B/A cases on the 36-response natural roster, stop and report generic ownership unresolved. Do not proceed to another native run under that architecture.

## Bottom line

The next smallest coherent move is **not** another prompt, threshold, or single-vector head. It is a value-masked, bundle-aware, token-level owner proposer with fixed graph compatibility and a hard natural owner evaluation gate. This directly targets the observed collapse modes while preserving the project constraints: no RAGTruth hallucination-label training, no fake natural-owner GT, no claim that retrieval equals truth, and no native mechanism claim before verified B/A exists.

References used: Khattab and Zaharia, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT, arXiv:2004.12832, §§3.2-3.3, https://arxiv.org/abs/2004.12832 and HTML mirror https://ar5iv.labs.arxiv.org/html/2004.12832.
