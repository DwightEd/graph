# Restoration v2 relation-representation review — 2026-09-13

Scope: bounded `research-refine` follow-up during the v2 run. I read the frozen representation path in `next_iteration/grounded_graph_features.py`, `next_iteration/grounded_graph_adapter.py`, `next_iteration/constraint_inventory.py`, the restoration-v2 method note, and v1 evaluation/erasure summaries. I inspected source inventory examples without hallucination labels or gold labels. I did not code, run GPU, or inspect natural answer correctness labels for this review.

## Direct answer

The current topology does **not** explicitly encode subject/record–predicate/field–value ownership as a relation topology available to the adapter. It encodes mostly containment and membership, with some ownership facts preserved as metadata outside the learned graph.

What is explicitly fed to the adapter:

- Node hidden states are pooled from original source prompt token spans.
- Node types are only the 9 coarse inventory kinds: literal field owner, source document owner, text context, surface component, word atom, field/record/weekday/context bundles.
- Edges are only four generic directions: component-to-parent, member-to-bundle, and their reverses.
- Message passing uses those generic edge-type embeddings plus node-type embeddings. It receives no field-key edge label, no key path embedding, no record role type, and no typed predicate relation.

What is preserved but not used as relation input:

- Data2txt fields carry `field_path`, `record_id`, `value_type`, `value_status`, and bundle `relation_label` in inventory JSON.
- `source_structure.edges` contain exact literal key spans such as `key_span` for `name`, `address`, `hours/Monday`, etc.
- Those key spans and path labels do not become source graph nodes or typed edges in `grounded_graph_features._source_nodes`; field nodes pool value-token spans, and bundle nodes pool member value/context/component spans.

A transformer value-state may implicitly contain nearby key information because it was computed in the full prompt. That is not the same as graph topology encoding ownership. It asks a small GNN to recover field identity indirectly from contextualized value vectors and generic bundle membership.

## Evidence from inspected examples

Data2txt source `14637` contains explicit source structure for the user’s hours/WiFi/rating failure family: fields such as `['hours','Monday'] = '8:0-16:0'`, `['attributes','WiFi'] = 'free'`, `['business_stars'] = 4.0`, and multiple `review_info/*/review_stars` fields. The inventory also has weekday bundle membership and record bundles. However, the model-facing edge kind for all these fields is still generic `member_of_provenance_bundle`; the relation label `hours / Monday` or `business_stars` is metadata, not an edge label or node text feature.

Data2txt source `13717` shows the same failure mode more starkly: address `220 W Carrillo St, Ste 1`, business rating `3.5`, review stars `4.0/5.0/4.0`, unknown `attributes/WiFi = None`, and long review text are all retained as fields. This fixes earlier coverage loss, but the adapter still has no typed distinction between `business_stars` and `review_stars` except what the frozen value states and generic record bundle can imply.

QA source `14308` and Summary source `11316` are represented as a document owner, sentence/context units, surface components, word atoms, and context bundles. There is no explicit semantic subject-predicate-value graph. This is honest as inventory, but it means the GNN cannot claim relation topology for natural textual facts beyond context containment.

## Interpretation of v1 and what v2 can test

The v1 natural result was negative: graph difference AUROC was 0.505927 on all words and 0.512073 post-first, while no-edge/permuted controls were comparable. The source-heldout erasure control showed coordinate pointer probability changed when source payload was erased, but generation gain did not become source-payload dependent in the intended direction. Those results are consistent with a model learning coordinate sensitivity without learning a useful natural ownership signal.

V2 is a valid single-variable test of the restoration path: it changes the query/decoder base to all-source-erased `H_empty` while retaining original source graph states. If it succeeds, it supports the claim that graph-source states can restore observed tokens beyond an empty-source base. If it fails to beat no-edge, it does **not** yet falsify the user’s desired “high-dimensional relation topology” idea, because the current graph has almost no typed relation topology to test.

If graph and no-edge remain tied, the most likely diagnosis is not that relation topology is useless; it is that the current relation topology is too weak to distinguish owner bindings such as business rating versus review rating, weekday bundle versus one weekday, or field value versus similarly typed scalar elsewhere in the same record.

## One minimal next structural change if v2 fails

Do one explicit typed-key/record/value binding test. Do not add a larger GNN, hallucination classifier, Qwen semantic templates, or a new free-form parser.

The minimal change is to materialize the source structure already present in inventory into model-facing graph nodes and typed relation edges:

1. Add `literal_key_node` for each `source_structure.literal_field` edge using its exact `key_span`. Its high-dimensional input is the frozen source prompt final-norm mean over the raw key token span.
2. Add `literal_record_node` for each literal dict/list record using the existing record raw span in `source_structure.nodes`; do not use absolute list-index embeddings.
3. Keep the existing value field node. Add one `field_binding_node` per observed field whose members are `(record_node, key_node, value_field_node)`.
4. Replace the generic-only field path with typed observed edges:
   - `record_has_key`
   - `key_points_to_value`
   - `value_in_record`
   - existing `field_has_component`
   - existing bundle membership for contexts/components
5. The edge type should be structural, while the key identity comes from the high-dimensional key node. Use only deterministic broad families when needed for grouping, such as `hours:weekday`, `attributes`, `business_stars`, `review_info:review_stars`, `review_info:review_text`; do not add source IDs, row IDs, or list indices as learned features.

The loss path stays small:

- Keep the same restoration CE through the frozen LM head.
- For copied source-coordinate anchor tokens, change the primary pointer positive from the bare value owner to the `field_binding_node` for literal fields. For long natural text, keep context/document positives; do not invent subject-predicate labels.
- Hard owner ambiguity is created naturally inside each source: same value across weekdays, same scalar type across ratings/dates, same city/name strings across records, and same field family across different records. No synthetic hallucination labels are needed.
- The no-edge arm must keep the same added nodes and parameters; only message edges are disabled. Otherwise a graph/no-edge gap could be capacity rather than topology.

This is still one adapter and one source-copy/restoration objective. The structural change is that graph topology finally carries the observed binding chain `record -> key -> value` instead of leaving key ownership as unconsumed metadata.

## Falsifiable natural evaluation

Freeze the structural variant before labels and report exactly the same natural dev metrics as v2:

- original LM NLL/entropy,
- no-edge restoration scores,
- typed-relation graph restoration scores,
- edge-permuted typed-relation graph scores,
- all-word and post-first AUROC/AUPRC with the same label-read boundary.

Add source-heldout, label-free owner diagnostics that target the observed failures:

- same value, different key: all weekdays with identical hours;
- same scalar type, different field: `business_stars` versus `review_info/*/review_stars`;
- same broad type, different record: review dates/stars across review records;
- unknown value owner retained: `attributes/WiFi = None` remains an owner candidate but not a factual truth label.

A credible positive result must beat both no-edge and edge-permutation on the natural frozen scores and improve the hard source-heldout owner diagnostics. If it only improves source-copy reconstruction but not natural detection, report the domain gap as unresolved. If it does not beat no-edge/permutation even on these owner diagnostics, remove or demote the GNN; use node-only attention/LM NLL as the simpler baseline and stop adding graph modules.

## Domain-gap boundary

This change addresses a specific gap: the current graph cannot directly represent field-key and record ownership, so failures like address number routed to ratings or business rating confused with review stars are structurally plausible. Materializing key and record nodes gives the model a learnable path for owner binding under the same no-hallucination-label contract.

It does not solve generic QA/Summary semantic ownership. For natural passages, the inventory still has only document/context/component containment unless a separate semantic relation extractor is introduced. The typed-key change can improve Data2txt ownership and some literal-source cases; it cannot by itself infer that a paraphrased summary clause requires a particular predicate, condition, or absent fact. If v2 or the typed-key variant fails mainly on QA/Summary while Data2txt owner diagnostics improve, the honest conclusion is partial: source-structure ownership helped structured sources, but the generic natural semantic provider remains unsolved.

## Required reporting language

Do not claim that v2 tested relation topology broadly. The precise claim should be: “v2 tested whether original high-dimensional source node states plus generic containment/member graph can restore observed tokens from an all-source-erased query.”

If a typed-key variant is run later, the claim can become: “observed source key/record/value topology improves or fails to improve source-coordinate restoration and natural error ranking under a no-hallucination-label training objective.” It still must not be called internal causal routing, abstract pointer binding, or full semantic truth detection without separate native and independent-label evidence.
