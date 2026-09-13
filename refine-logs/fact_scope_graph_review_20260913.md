# Fact-scope graph revision review — 2026-09-13

## Verdict

The revision addresses the observed failure at the correct layer: the prior surface pipeline had no valid complete B/A contrasts, so expanding native interventions would have been a category error. An overlapping, non-contiguous fact-incidence representation is preferable to mutually exclusive word windows. It is also a sensible use of shared variables: shared text need not share truth, while an explicitly verified identity can constrain compatible source records.

This is not a return to the earlier free SRL design **if** factor construction remains a finite, span-bound, abstaining interface. Four Required contracts below are needed before the new representation/solver can feed a native runner. They are small interface restrictions, not a request for another unsupervised extractor, GNN, toy experiment, or gold labels.

The literature supports the representation boundary, not automatic semantic correctness. PropSegmEnt represents propositions as token subsets, explicitly allowing shared predicate/argument material, but uses expert annotations; it does not establish that a CSP can infer propositions from arbitrary generated text. [PropSegmEnt](https://aclanthology.org/2023.findings-acl.565.pdf) GraphFC similarly uses relational constraints to reduce mention ambiguity, but its graph construction and checking are components to validate, not a proof that graph topology supplies truth. [GraphFC](https://arxiv.org/abs/2503.07282) Claimify's ambiguity handling likewise supports abstention rather than forced decontextualization. [Claimify](https://www.microsoft.com/en-us/research/blog/claimify-extracting-high-quality-claims-from-language-model-outputs/)

There is no mechanism-effect result here: surface v1 has zero eligible contrasts, the replacement readout is not yet validated, and MLP aggregation remains unresolved.

## Required 1 — factor construction must be a finite span protocol, not renamed SRL

Every FactFactor must be sealed as a provenance object before any source candidate, CSP solve, feature capture, or native search. The minimum object is:

    factor_id, response/base identity and text SHA
    member_spans: exact, ordered character/token members M_f
    target_members: exact complete semantic answer members
    non_target_constraint_members: exact spans and roles
    relation_template_id and renderer version
    source-window / response-prefix boundaries
    status: asserted | nonassertive | ambiguous | unsupported
    raw reader request/response receipt, schema and model identity

The displayed paraphrase is explanatory only. It cannot be an input to native alignment or a replacement for M_f. A factor must be rejected if its target is a function-word fragment, a partial range/unit/quote, has a missing required member, crosses an unsealed source/response boundary, or is not a response assertion.

The only permitted semantic operation is a constrained selection over program-provided span IDs and a fixed, versioned relation-template inventory; each selection can return unknown/ambiguous. The reader must not invent a subject, predicate, condition, source record, or rewritten claim string. Its answer must identify exact existing member IDs and an explicit unknown status. This is materially narrower than free SRL: the model cannot create nodes or edges, and all incidence is checkable against raw text.

Because the existing first-token semantic interface has already failed in practice, the new relation reader needs a CPU observability gate before it is used: actual rendered requests must have a valid label/token map, non-degenerate finite distribution, stable parser, exact receipts and prompt/model/code hashes. Failure returns relation_reader_unobservable and leaves the full factor denominator; there is no lexical/top-1 fallback that silently becomes an owner edge.

## Required 2 — leave-current-value-out must cover every owner-solver payload

The document correctly leaves the challenged relation value out of owner resolution, but this needs a mechanical payload boundary.

Split two receipts:

1. **Response-fidelity receipt** may view the original base and establish only that the response asserted the marked factor/target members.
2. **Owner/constraint receipt** receives the same marked relation with target members replaced by a typed placeholder, and source candidate views with the candidate answer field/value masked. It receives source parent/record identity, field paths, non-target constraints, and exact span IDs, but not the original target display/value, target-derived hidden state, owner ranking, native effect, or labels.

The compiler should fail if the original target-display bytes occur in any owner-solver request field other than the isolated response-fidelity receipt. This includes explanation strings and prompt templates, not just a top-level payload key. A source may naturally contain another identical literal, but it must be represented as a distinct masked candidate field, never as a value match supplied to the solver.

This restriction prevents a value-conditioned answer from reappearing through a graph explanation and recreating the old same-type/same-value owner shortcut. It does not prevent later full-source finite verification or the separately frozen source-competitor roster from using the actual literal for their defined estimands.

## Required 3 — three-valued, scope-limited CSP semantics

The CSP may combine certified compatibility facts; it must never convert prose or graph adjacency into an allow table. Each candidate relation constraint needs one of:

    allowed          independently supported for this exact variable binding
    forbidden        independently contradicted for this exact variable binding
    unknown          ambiguous, reader-unobservable, no source scope, or unsearched

Unknown is neither allowed nor forbidden for proving uniqueness. A solver result is:

- owner_certified only when every in-scope non-target constraint needed for that binding is allowed and every remaining owner is forbidden by an independently certified constraint;
- owner_ambiguous when two or more assignments remain possible, or any necessary exclusion is unknown;
- constraint_conflict only when certified constraints in the **same factor scope** are jointly inconsistent;
- search_incomplete when the frozen bounded enumeration cannot establish either result.

A non-target response error must not make the solver select a different “closest” source owner. If it is a condition, subject, time, unit, negation, quantifier, or event identity required by f, it can disqualify a candidate binding or make the factor unresolved. If it belongs to a separate co-occurring factor, it must not enter f's CSP at all. Likewise, repeated text may share z_v only after an exact-span identity/coreference constraint has been independently certified; entity-string equality, same parent sentence, adjacency, topic, or source record are insufficient.

This keeps a bad non-target assertion from either leaking its value into the challenged target or forcing a wrong owner through over-wide constraint sharing. It deliberately reduces coverage. The alternative—treating unknown as permissive, or using an unverified constraint to break a tie—would make the graph a renamed free semantic extractor.

## Required 4 — define the joint-search / single-layer-route handoff before effects

The all-shared-prefix-query × all-layer search is executable with the existing CausalOracle only as a **joint position/content effect**: multi-layer gates are applied together, so downstream layers correctly recompute and their current values/masses may change. Preserve the complete searched tree, joint groups, invalid groups, unsearched tail and native calls. A joint effect must not be decomposed into independent layer effects.

The B→A versus B→U1/U2 routing certificate is separately executable only under the one-way contract already reviewed:

- choose one layer L and query set Q before any B→destination outcome, from a frozen candidate-blind rule or a frozen joint-search record;
- run every B→A/U branch from the same ungated branch baseline, with the identical B selected keys, source key domain, L, Q and strength;
- pre-freeze A and U controls with per-head/query receiving-mass and visibility matching at L×Q; reject zero receiving mass, all-future/no-op geometry, future-weight violations and missing controls;
- retain only the existing claim: same B current outgoing mass, destination changes. Do not call it a joint-layer transport result.

If the all-layer/group search succeeds but no independently predeclared single layer has a valid B→A geometry, emit joint_position_effect_single_layer_route_unresolved. Do not search layers after seeing repair effects, and do not apply the single-layer equal-mass language to a multi-layer intervention. This is a coverage abstention, not evidence against source routing.

## What can be claimed after these changes

A complete strict B/A contrast can support only the following ordered statements:

1. A span-bound factor was asserted and its target relation has a source-supported, owner-certified alternative under preserved in-scope constraints.
2. A raw original-source input path mediated an effect through a fixed V receiver address.
3. Relocating the same current mass from a finite-I, wrong-occurrence B to correct A repairs D_f selectively against two frozen inapplicable destinations.
4. A separately measured history path may coexist; failed history measurement is unresolved, not excluded.

Steps 1–3 support a **conditional wrong-source-occurrence routing witness**. They do not identify an abstract pointer, prove that relation variables are internally represented, show a global factuality decision, or establish a continuation-error graph.

The current evidence×MLP four-state test remains a modulation diagnostic. A stronger “correct information arrived but aggregation failed” statement needs an intervention that independently controls the already-certified arriving message and a later aggregation operation, repairs the same D_f, and remains selective against frozen controls. No such natural result exists in this design; label it aggregation_unresolved rather than assigning every route failure to MLP.

## Minimal validation and denominators

Before any GPU native stage, report on all 36 frozen responses:

- response tokens covered by one or more factor members, nonassertive/ambiguous/uncovered tokens, and overlapping membership count;
- factors attempted, reader-unobservable, malformed, owner_ambiguous, conflict, search_incomplete and complete contrast counts;
- all roster/candidate/control rejections without backfill;
- actual CPU reader calls and the sealed graph/solver/code/model inputs.

After valid contrasts exist, compare the graph solver against the same finite reader without shared-variable constraints and an explicit no-CSP abstaining baseline. The comparison concerns candidate/contrast coverage and disagreement states; it is not a ground-truth accuracy result. Keep official labels completely outside preparation/native phases and add them only after complete prediction artifacts are frozen.

## Required changes summary

1. Make relation factors constrained span selections with a fail-closed reader-observability gate.
2. Enforce a byte-level target-value exclusion boundary for all owner-solver payloads.
3. Use scope-limited three-valued CSP certificates; unknown cannot establish a unique owner.
4. Freeze the joint-position to single-layer-route handoff and downgrade absent single-layer geometry to unresolved.

With these changes, the proposal stays bounded and addresses the concrete zero-contrast failure without relabeling free extraction, generic graph effects, or MLP interactions as a mechanism result.


## Targeted revision — executable relation language, coordinate masking, and reader boundary

This addendum supersedes the fixed-template and whole-payload substring language in Required 1 and Required 2. The two objections are correct.

### 1. Factor language need not be a closed relation inventory

A closed relation-template inventory is not necessary and would recreate zero coverage for natural, previously unseen relations. The safe boundary is instead:

- a factor carries an exact **relation footprint**: ordered, existing response spans that express its predicate/event/condition structure, plus exact target and non-target member spans;
- the reader may reason over the untouched original sentence/base and its prior context, including arbitrary natural-language relation wording;
- its output is constrained to existing response/source span IDs, occurrence/record IDs, an explicit relation-footprint subset, and unknown/ambiguous. It cannot introduce a new predicate, entity, condition, event, source record, or a rewritten fact whose member spans do not exist;
- any explanatory paraphrase remains display-only and cannot be consumed by native alignment, candidate construction, or the CSP.

Thus the graph can express an unseen relation through original text rather than a schema label, while every claimed incidence/variable binding remains auditable. A factor with incomplete members, a target that is only a function-word fragment, or an output referring outside the sealed span universe still fails closed. This is a span-constrained semantic readout, not free SRL.

### 2. Replace byte-substring banning with occurrence-aware masking

A raw byte ban is indeed unsound: the target "1" can occur in IDs and the target "on" can be an unrelated condition. Owner inputs need coordinate- and role-aware views, not global replacement.

For every owner solve, freeze a masking manifest keyed by document SHA and exact spans/field paths:

    target_ref = (response/base SHA, target member spans, semantic dependency footprint)
    candidate_value_refs = exact value spans for every source occurrence under comparison
    same_value_occurrence_refs = every separately identified value-bearing occurrence
    retained_constraint_refs = all non-target factor members and their roles

The owner view replaces only target_ref with a typed target marker, and masks each candidate's designated value-bearing source span with an occurrence-specific typed marker. Other matching literals are not silently deleted: each must be classified by coordinates as (a) another value-bearing occurrence, included in same_value_occurrence_refs and masked in the comparable owner view, or (b) a non-value occurrence such as an ID/condition token, retained with its role in retained_constraint_refs. The manifest and rendered view must round-trip to the original text by coordinates; any overlap/role ambiguity returns mask_scope_ambiguous.

This prevents the current target value from selecting an owner through its own answer position while preserving genuine non-target constraints. It also makes same-valued records visible as distinct candidate IDs rather than accidental lexical evidence. Full values may reappear only in later, separately sealed source-support and B/A verification requests, never in the owner-solver decision payload.

### 3. First-token observability is a transport diagnostic, not a semantic gate

The existing CPU result—well-formed letter outputs with near-unit mass but invalid semantics—shows that a first-token label map/parser check cannot be a relation-semantic acceptance gate. It should remain a required **transport diagnostic** for any finite interface that uses labels: validate request rendering, token mapping, parser, finite scores, caching and receipt identity. Passing it says only that the interface executed.

The new relation readout may use a different frozen query/reasoning protocol, including natural original relation wording and a structured span/citation response. Its semantic contribution is limited by the span-bound output above and is assessed by downstream consistency:

- response-fidelity must identify the asserted target/footprint in original text;
- owner solving can use only explicit per-tuple allowed/forbidden/unknown results;
- a native-eligible contrast still requires the later complete source-supported alternative and preserved-scope B/A checks.

No formatter success, confident label, or untrained reader score alone certifies a factor, owner, or source relation. If the new protocol is semantically inconsistent, it must produce unknown/ambiguous and retain the denominator; this is a method result, not a reason to reintroduce lexical top-1 fallback.

The remaining Required 3 and Required 4 are unchanged: per-tuple unknown cannot prove exclusion, factor scope limits non-target constraints, and a joint all-layer position result must hand off to a separately pre-frozen single-layer B→A/U route measurement or be reported route-unresolved.

