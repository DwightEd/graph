# graph boundary revision review — 2026-09-13

Scope: bounded method-design review only. I did not modify the frozen `soft_graph_v1` code or start GPU work. I read `refine-logs/segmentation_architecture_lit_20260913.md`, current `soft_graph_structure.response_claims`, `source_event_graph.raw_inventory/compile_pointer_events`, and local prior structural notes. The concrete failures considered are: a 128-token lexical inventory boundary splitting a sentence at “digestive | and respiratory systems”; an incomplete Qwen anchor ending at “in a” before “1,350-year-old ...”; and quote/punctuation-only leaves.

## Verdict

The next revision should change the segmentation contract, not add another extractor or GNN. The current bug is architectural: three different objects are being treated as semantic cut points:

1. raw token inventory units, which exist only to address text and bound model prompts;
2. `text_units` punctuation/newline spans, which are weak formatting spans;
3. Qwen pointer-event anchors, which can be incomplete or malformed even when coordinate-valid.

None of these should be allowed to define a factual claim boundary by itself. The minimum repair is a two-layer boundary interface:

```text
RawInventory(address-only chunks; never semantic)
  -> BaseUnits(complete sentence or program-enumerated complete clause; indivisible)
  -> AnchorProposals(nested Qwen/pointer roles; advisory, not boundary-setting)
  -> BoundaryGraph(base-unit nodes + role/source/history/native edges)
  -> SpanRegions(adaptive connected/contiguous groups; scope-labeled)
```

This keeps all original token coordinates and cross-span edges, but prevents address chunks or incomplete anchors from becoming semantic claims.

## What the local literature implies

The segmentation notes support a conservative design:

- StructScore uses RST/source blocks and fallback windows to aggregate NLI-like scores. That supports hierarchical blocks as organization, but not treating a fixed window as a factual unit or internal route boundary.
- Structured Attention Networks and segmental RNNs show how to learn latent chains/segments, but both need a downstream target. Without a reliable boundary objective, a latent segment layer can learn length/format shortcuts. It should not be introduced here under the “no hallucination-label training” constraint.
- SIRG-style semantic internal graphs show why adaptive edge selection over token/semantic nodes is useful, but their final detection uses supervised AlignScore-style discrimination and aggregation choices. We can borrow the separation between low-level token flow and semantic group nodes, not the supervised detector or a max-aggregation truth claim.

The resulting design should use deterministic base units plus fixed graph aggregation. If labels are not used, do not train a boundary model and do not call the boundary “learned factual span.”

## Minimal boundary interface

### 1. Raw address units are never claim boundaries

`raw_inventory(unit_tokens=128)` can stay as an addressing and request batching layer. Its unit spans must not be inserted into semantic `bounds`. A source/response sentence may cross several raw units; that is a request-scheduling issue, not a semantic split.

If a reader can only process a window, the request envelope should include:

- `address_window_id`, `address_window_span`, and token indices local to that window;
- the enclosing base sentence/clause ID and global character offsets;
- a rule that returned anchors are projected into global coordinates and then attached to the enclosing base unit.

A raw-unit boundary inside a base unit must produce at most `reader_window_boundary_inside_base=true`; it must not produce two claims.

This directly fixes the 13803 failure: “between the digestive and respiratory systems” remains one base unit or one clause because the 128-token address chunk is ignored as a semantic boundary.

### 2. Base units are complete sentences or safe clauses

Create `BaseUnits` before using Qwen anchors. A base unit must contain at least one lexical content token and must absorb trailing quotes/closing punctuation into the neighboring content span. Quote-only or punctuation-only spans are not claims.

Recommended base construction:

1. Build sentence envelopes by punctuation/newline with quote/bracket balancing and attachment. A terminal quote, comma, period, parenthesis, or standalone discourse marker cannot be emitted as a separate base.
2. Within each sentence, enumerate clause candidates only when a deterministic split is safe: semicolon/colon/dash, or comma/coordinator/subordinator where both sides contain an independent predicate cue or a validated pointer event anchor with subject+predicate.
3. Do not split noun/list coordination without a second predicate. In “digestive and respiratory systems,” `and` is inside one role/list, not a clause boundary.
4. If the safe-clause test fails, keep the whole sentence as the indivisible base.
5. For very long sentences, keep the base span intact and split only the reader request windows; if a base exceeds the native/semantic budget, mark `base_over_budget` and keep it in the denominator rather than slicing it.

A base unit schema should be explicit:

```json
{
  "base_id": "...",
  "kind": "sentence|clause",
  "span": [start, end],
  "text": "exact original substring",
  "sentence_span": [start, end],
  "address_unit_ids": ["u0", "u1"],
  "split_basis": "sentence_terminal|safe_clause|unsplit_sentence",
  "content_token_count": 12,
  "indivisible_for_semantic_scoring": true
}
```

### 3. Qwen/pointer anchors are nested proposals, not leaf spans

Compile Qwen pointer output as now, but assign every retained event/role to the smallest enclosing base unit. Anchor endpoints should not enter the base boundary set. A coordinate-valid anchor can still be semantically incomplete.

Add anchor quality flags:

- `anchor_inside_base=true|false`;
- `anchor_coverage_ratio` over the base’s non-punctuation tokens;
- `right_open_edge` if the anchor ends on a determiner/preposition/conjunction/comma-open fragment or leaves immediately following numeric/nominal material;
- `left_open_edge` if it begins after an unresolved modifier/linker;
- `unassigned_content_spans` from `compile_pointer_events`, classified as content vs punctuation;
- `anchor_status=complete_event_candidate|partial_event_candidate|truncated_anchor|nonassertive_anchor`.

A `truncated_anchor` may supply role candidates if its role pointers are valid, but it cannot shrink the claim. This directly fixes the 2913 failure: “Archaeologists have discovered a stone chest in a” remains an incomplete proposal under the larger sentence/clause base; the next “1,350-year-old ...” material is not detached into another claim merely because the anchor stopped early.

Quote-only and punctuation-only anchors become `nonassertive_anchor` and cannot create claim leaves. This fixes the 5153 class.

### 4. Semantic scoring uses base units; role localization remains nested

Run full-source SCNU and soft graph risk on base units, not on raw chunks or incomplete anchors. The verifier receives the base text and earlier response prefix up to the base end. For clause bases, later clauses are not used as evidence; if a later phrase is required to interpret the current text, the safe-clause splitter should have kept the whole sentence together.

Nested role slots still exist:

- A parsed role inside a base can get source candidates, local SCIU, and source-pointer bridge attempts.
- Slot-level B/A certificates are allowed only when the role pointer, non-target conditions, and deterministic edit pass.
- If slot checks fail but the base unit is factual, the output remains base-level `claim_scope`, not punctuation/anchor-level `slot_scope`.

This is the key compromise: local role slots can be independently localized when evidence is strong, but incomplete Qwen anchors no longer decide the semantic span.

## Boundary graph aggregation

After base units are created, build an adaptive graph over base units and nested role/event nodes. No hallucination labels or trainable boundary head are needed for the first revision.

### Nodes

- base units: sentence/clause spans;
- nested pointer events and roles;
- source nodes/fields selected by matcher;
- history base units and prior role/event nodes;
- native groups/certificates when available.

### Edges

Use typed fixed edges with recorded evidence:

- adjacency edge between neighboring base units;
- same source record/event/source-node edge from matcher marginals and finite relation verification;
- same response event edge when multiple anchors live under the same base;
- history reuse/correction/new-topic edge from finite history classifier;
- native target-dependence edge from soft graph;
- strict A/B source/history origin edge when bridge certificates pass.

### Merge rules

Do not merge merely because of proximity or punctuation. Merge adjacent base units into a continuous region only when at least one positive relation edge exists and no strong split edge exists:

- positive relation edge: same validated source owner/event, finite history reuse `R>=0.8`, certified origin-mediated continuation, or same parsed event spanning both clauses;
- split edge: finite correction/new-topic `>=0.8`, different validated owner/event with incompatible predicate/condition, or base extraction marked `base_over_budget/unresolved` without relation evidence.

Non-adjacent history/source edges remain in the graph, but word-level continuous spans should stay contiguous unless the output explicitly reports a noncontiguous graph region. This avoids painting unmeasured middle text as erroneous.

For scoring, report both:

- `base_risk`: the soft graph risk for each indivisible base;
- `region_risk`: an aggregation over connected base units with the same typed relation basis, e.g. max risk plus edge reasons, not a learned or calibrated posterior.

## Can this distinguish the hard cases?

### Sentence-internal owner switch

Yes, but only when the safe clause or nested pointer evidence exposes two events. A sentence like “Alice won in 2019, and Bob won in 2020” can split into two base clauses because both sides have predicate cues/anchors. A sentence like “between the digestive and respiratory systems” must not split because the conjunction is inside one role/list. If the owner switch is implicit and no safe clause/anchor exists, the base remains whole and the output scope is `owner_switch_unresolved`, not a fake local span.

### Continuous error range

Partially. The graph can group consecutive base units when there is a typed reuse/source-origin relation and consistent risk direction. This is enough to report `reuse_connected_region` or `origin_mediated_continuation_region`. It cannot prove the exact first/last erroneous token without independent evidence. If the middle units are unmeasured or the relation edges skip over them, output a noncontiguous graph region or an unresolved boundary rather than a continuous span.

### Local role-slot定位

Yes for role pointers that are nested inside a valid base and pass local source relation / bridge checks. Otherwise the unit of factual scoring is the base sentence/clause. This prevents role-level overclaiming when Qwen gives an incomplete anchor or misses a condition.

## Output contract

Each response should include four denominator layers:

1. `raw_address_coverage`: all characters/tokens and address windows, explicitly not semantic.
2. `base_units`: all sentence/clause base spans, with content-token counts and split basis.
3. `anchor_proposals`: all Qwen/pointer events nested under bases, with completeness/truncation/nonassertive flags.
4. `boundary_regions`: graph-adaptive connected/contiguous regions with edge reasons, unresolved edges, and risk aggregation.

Every word maps to exactly one base unit. A word may map to multiple nested anchors/roles, but an anchor cannot remove it from the base denominator. Punctuation-only material is attached to neighboring bases or marked nonassertive, never emitted as a factual claim.

Failure statuses to record:

- `address_boundary_inside_base`;
- `truncated_anchor_under_base`;
- `nonassertive_anchor`;
- `safe_clause_split_unavailable`;
- `owner_switch_unresolved`;
- `condition_residue_unresolved`;
- `base_over_budget`;
- `region_edge_unresolved`.

## Minimal tests before replacing v1 segmentation

Use CPU-only unit tests; no GPU is needed.

1. A sentence crossing a 128-token raw inventory boundary yields one base unit or safe clauses, never two claims at the raw boundary.
2. “between the digestive and respiratory systems” is not split at `and`.
3. A pointer anchor ending at “in a” is flagged `truncated_anchor_under_base` and cannot define a leaf; the enclosing sentence/clause remains the claim scope.
4. Quote-only and punctuation-only spans do not become base claims.
5. A two-clause sentence with two explicit subjects/predicates can produce two safe clause bases.
6. Every nonspace word belongs to exactly one base unit, even if no pointer event is retained.
7. Nested role slots can be attached to a base without altering its boundary.
8. A history edge may connect non-adjacent base units, but continuous word spans are emitted only for adjacent connected bases.

## What not to do

Do not add a learned semi-CRF, GNN boundary head, or supervised detector under the current no-hallucination-label-training anchor. Do not let Qwen anchor completeness act as a hard boundary. Do not treat raw address windows, punctuation leaves, or request chunk limits as semantic spans. Do not make another source-QA prompt the gating fix.

## Required design change for next version

Replace `response_claims` with a base-unit compiler whose boundaries come only from complete sentence/safe-clause units. Pointer/Qwen events become nested advisory proposals. The downstream soft graph and strict bridge should operate at base scope first, then optionally refine to role slots when the nested role and finite relation checks pass. This is the smallest change that addresses the observed natural failures while preserving the user’s requirements: graph-adaptive spans, full context, cross-span edges, no hallucination-label training, and no fake semantic boundaries from fixed addressing units.

## Addendum: word-level mapping must not inherit whole-base bag risk

The base-unit repair above should not be read as “score the whole sentence as erroneous.” It fixes context and anchor completeness; it does not by itself localize every false word. The next interface must carry three scopes separately:

1. **Readable context anchor**: complete sentence or safe clause base. This is the minimum text shown to finite verifiers and native contrast compilers so that conditions, owners, and syntax are not cut off.
2. **Target role / factual span**: a nested role, value, condition, or deterministic bridge target inside the base. This is the only scope that may receive slot-level error localization.
3. **Graph reuse / propagation region**: connected base units linked by reuse/source/history/native edges. This is an influence or continuation region, not a token-level factual-error mask.

`GLOBAL_PROMPT` SCNU over a base is bag-level evidence: “at least one factual assertion in this base is unsupported/contradicted/unknown.” It is not evidence that every word in the base is wrong. Therefore `words_with_scores` should stop assigning a base claim’s risk to every overlapping word unless the claim has an explicit localizable span.

### Minimal implementable word mapping

For each word token, output separate fields rather than one max-overlap risk:

```json
{
  "word_span": [a, b],
  "base_id": "...",
  "base_bag_risk": 0.83,
  "base_bag_label": "C|N|U|S|mixed",
  "localized_risk": 0.5,
  "localized_scope": "target_role|fact_span|unknown_within_base|nonassertive",
  "localized_span_ids": [],
  "region_ids": ["..."],
  "region_scope": "reuse_connected_region|origin_mediated_continuation|none"
}
```

Rules:

- If a strict bridge single-slot `B(C/N)->A(S)` passes, assign the direction/risk only to the target role span. Non-target words in the same base keep `localized_risk=0.5` and `localized_scope=unknown_within_base`, while the base still carries `base_bag_risk`.
- If a finite local verifier identifies a specific role/condition as unsupported but no strict B/A certificate passes, assign only `localized_scope=fact_span` with `certificate=false`; do not spread to the sentence.
- If only full-source SCNU says the base is C/N and no role/fact span is identified, leave all words `localized_risk=0.5`, set `localized_scope=unknown_within_base`, and report the base-level bag risk separately.
- If multiple local spans overlap, combine only over those explicit spans; do not use the enclosing base as a fallback span.
- A graph reuse/propagation region can attach `region_ids` to words for visualization, but it must not change `localized_risk` unless the word also lies in a certified/localized target span.
- Supported or nonfactual base decisions may lower/report base-level risk, but they still should not be used as token-level correctness labels unless every factual role span in the base has been explicitly covered.
- Punctuation/quote-only material inherits `base_id` for rendering and context, but uses `localized_scope=nonassertive`.

This rule is label-free: it uses only predeclared spans from base compilation, nested role pointers, finite verifier outputs, and strict bridge certificates. It does not tune span expansion against RAGTruth or any other labels. If token-level evaluation is required, report two curves/tables separately: base bag detection and localized token/span detection. Do not use the stronger base bag score as the token score, because that would reward overbroad spans and recreate the original collapse in the opposite direction.
