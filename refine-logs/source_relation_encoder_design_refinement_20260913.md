# SourceRel targeted refinement — 2026-09-13

Scope: targeted `research-refine` refinement of `source_relation_encoder_design_20260913.md` before implementation. This responds to four concrete concerns: target-surface masking, pair-label tautology, overbuilt losses, and response-side parser dependence. I used the existing local notes on PropSegmEnt, Claimify, GraphFC, LLM2Vec/Qwen embedding, `semantic_provider_decision_20260913.md`, and `source_self_supervised_owner_review_20260913.md`. I did not browse, run GPU, read labels, or edit implementation.

## Revised decision

Build **SourceRel-Mini**, not the broader previous SourceRel. Its only trained claim is source-side relation ownership retrieval:

> Given a parser-light masked response context and a frozen source graph, rank reusable source occurrence/event owners and output a soft relation-role proposal. It is trained from source structures only, using masked source contexts and held-out relation reconstruction, with no hallucination labels and no free-text Qwen truth gate.

It does not output support/conflict/N. It does not replace typed comparators, finite relation verification, or native certificates. Its success criterion is candidate supply and owner/condition disambiguation under source-heldout tests.

## Correction 1: mask dependency closure, not every matching surface

The earlier blanket ban on target surface / number / unit was too broad. It would delete independent non-target homographs that are exactly needed to test owner binding. The correct rule is graph-dependent masking.

For a source training target node `t`, define a mask closure:

```text
D(t) = {t}
     ∪ same_variable(t)
     ∪ alias_of_same_role_value(t)
     ∪ duplicate_mentions_proven_same_fact(t)
     ∪ interval_endpoint_aliases_for_the_same_endpoint(t)
```

Only nodes in `D(t)` are masked. A same string, same number, same unit, or same name outside `D(t)` remains visible and is flagged as a hard-negative homograph. Examples:

- If closing time `16:0` appears as the same endpoint of the same all-week hours fact, mask it.
- If `4.0` appears as a business rating while the target is a review rating or time endpoint, keep it and mark `same_surface_different_relation`.
- If a business name appears as the owner context while the target is an address/city/time, keep it.
- If the hidden target is a time endpoint, keep the coordinate/type information such as `<TIME_RANGE_END>` but hide the value-bearing text for that endpoint. Independent times elsewhere in the parent context remain.

This preserves coordinate/role-aware masking while preventing direct answer leakage. The required evaluation is not a global string-ban test; it is a **heldout target-value dependency test**: performance must not collapse when the same target value appears only in unrelated homograph contexts, and the model must not choose those homographs as owners.

## Correction 2: topology is fixed evidence unless the predicted edge is actually hidden

A pair label like `same_record` or `same_event` is tautological if the scorer receives parent IDs, direct relation edges, or raw parent spans that include both candidate nodes. The pair objective is only meaningful under a masked-edge view.

SourceRel-Mini therefore uses topology in two different ways:

1. **Fixed graph factor** at inference and training-pool construction. Known source graph edges define positives, hard negatives, mask closures, multi-positive equivalence, and event-beam compatibility. This is not learned evidence; it is provenance.
2. **Optional learned pair regularizer** only when the direct relation is removed from the input graph. If this cannot be enforced, skip the pair loss rather than pretending to learn graph structure.

The pair reconstruction input for nodes `(a,b)` must remove:

- the target edge family being predicted between `a` and `b`;
- parent/event/record ID equality features for that pair;
- direct `same_record`, `same_event`, `same_field`, `same_variable` booleans;
- raw parent text windows that contain both nodes together, unless the counterpart span is masked;
- absolute char/token positions and source-local node IDs.

Allowed inputs are local masked node views, value/field types, field-path suffix tokens, neighboring edges not in the held-out target family, and separate parent summaries that do not reveal shared identity by ID.

Target relation families should be coarse and useful for assignment:

```text
same_event_or_record_role_bundle
same_condition_scope
different_condition_same_owner
same_value_different_owner
field_sibling_compatible
no_compatible_relation
```

If a no-text/ID baseline or direct graph lookup solves the pair task, that task is invalid as evidence. The paper should then report topology as a fixed factor, not a learned pair mechanism.

## Correction 3: reduce the loss to two terms

The previous five-loss recipe was overbuilt before any candidate-supply benefit is shown. SourceRel-Mini has only two losses:

### Loss 1: multi-positive owner InfoNCE

For a masked source query `q` and candidate pool `C(q)`:

```text
L_owner(q) = -log Σ_{p∈P(q)} exp s(q,p)
                  / Σ_{c∈C(q)} exp s(q,c)
```

`P(q)` contains exact source occurrence(s) proven by the source graph to fill the hidden role. Same-value different-event candidates are negatives unless the graph proves they are duplicate mentions of the same fact.

### Loss 2: masked-edge relation reconstruction

Only if the held-out edge masking above is implemented:

```text
L_pair(a,b) = CE( ψ(masked_view(a,b)), ρ_ab )
```

This trains relation compatibility without exposing the relation label as an input. It is an auxiliary representation objective, not a truth verifier.

No margin loss, entropy regularizer, or learned NULL deletion objective in v0. `NULL` / `UNKNOWN` at inference are status outcomes from coverage, low score, near tie, or missing source graph coverage. They are not factual N and they are not trained as not-stated labels. A deletion/null objective can be added later only if candidate supply works and matched deletion pools can be constructed without count/type/coverage leakage.

## Minimal model

### Inputs

Source graph `G_s`:

```text
nodes: source occurrence, parent sentence/clause/event, record/field, condition, shared variable
edges: contains, field_child, event_role, condition_of, same_variable, list_member, temporal_scope, negation_scope
```

Response query uses no free SRL. It is built from exact surface proposals and a whole readable parent context:

```json
{
  "parent_text": "full sentence or clause envelope",
  "target_span": [start, end],
  "target_surface_type": "time_range|number|date|content_run|quote|...",
  "masked_parent_text": "same parent with target replaced by typed mask",
  "non_target_context": "all other original text retained",
  "coordinate_features": ["range_endpoint=end", "inside_hours_phrase", "quoted=false"],
  "source_task_family": "Data2txt|QA|Summary|unknown"
}
```

The target value text is masked for owner scoring. The model may know broad type and coordinate, because those are available without knowing the correct source owner. It may not use the target value to pick a source candidate. Value comparison happens later in typed/finite verification.

### Encoders and score

Use frozen text features plus a small trainable scorer:

```text
q = pool(Encoder(masked_parent_context, target mask coordinate))
c = pool(Encoder(candidate occurrence + candidate local source context))
φ = typed/provenance features without direct answer or pair-label leakage

s(q,c) = (P_type q)^T (Q_type c) / τ + w^T φ(q,c)
```

The model also outputs a soft relation-role distribution over source-derived role families:

```text
p(role_family | q) = softmax(R q)
```

This role distribution is trained implicitly through the owner positives or with the same owner loss decomposition; do not add a separate third loss unless the implementation cannot expose relation-role scores otherwise.

### Event beam with reusable source facts

For multiple target surfaces in one response parent, combine unary owner beams with fixed source graph compatibility:

```text
E(z) = Σ_r -s(q_r, z_r)
     + β Σ_(r,r') incompat_fixed(z_r, z_r', response_parent_scope)
```

There is no one-to-one source capacity. A source fact may support multiple response mentions. The compatibility factor can prefer same source event/record/condition for targets in one parent context, but it cannot force one source event when relation-role scores are ambiguous. It should output near-tie/ambiguous rather than invent event identity.

## What topology adds over lookup

Topology is useful only in four concrete places:

1. It defines mask dependency closure and prevents same-value leakage from true duplicate mentions.
2. It creates hard negatives that a lexical matcher would confuse: same value different event, same field different record, same owner different condition, same predicate different object.
3. It supplies fixed event/condition compatibility for multi-role assignment while allowing source reuse.
4. It enables validation splits that expose field-path memorization and position shortcuts.

If Data2txt typed rules already map a response to a field exactly, use the typed rule. SourceRel-Mini is for cases where lookup cannot decide owner/condition from response context but a source-derived relation model might supply a candidate beam.

## Heldout validation before implementation is trusted

Required source-only validation:

- `source-heldout owner top-k recall`: no shared source IDs between train and validation.
- `target-value dependency heldout`: hidden values and homograph contexts partitioned so exact value reuse cannot explain success.
- `same-value wrong-owner rate`: must beat a lexical/value matcher by a large margin.
- `same-field other-record error`: Data2txt field-key memorization check.
- `same-owner different-condition error`: time/stage/negation/quantifier disambiguation.
- `masked-edge pair accuracy`: only counted when direct relation edges/IDs/parent co-text are removed from input.
- `position-shuffle robustness`: source node order and absolute positions randomized/canonicalized.
- `no-graph baseline`: compare against frozen hidden bi-encoder without graph hard negatives/fixed compatibility.
- `rule-matcher fallback comparison`: SourceRel must increase finite-verifier candidate supply over deterministic typed/rule matching, not just add uncertain candidates.

Natural RAGTruth labels can evaluate final detection later, but cannot select the checkpoint, tune thresholds, or define hard negatives.

## Failure bounds

Reject or downgrade SourceRel-Mini if any of these occur:

- owner recall comes from exact value/string matching under homograph tests;
- pair reconstruction is solved by exposed parent IDs, direct edges, or shared raw parent text;
- field/path heldout collapses, showing lookup memorization rather than relation ownership;
- response parser-light query lacks enough context to identify owner/condition and outputs only near ties;
- relation verifier still rejects most top-k candidates, so SourceRel only increases unverified noise;
- NULL/UNKNOWN is interpreted as not-stated or factual N;
- source graph extraction misses the relevant source event, because the encoder cannot recover facts absent from its candidate graph.

## Implementation recommendation

Implement only this minimal sequence:

1. Build source-only masked training instances with dependency-closure masking and homograph flags.
2. Train owner InfoNCE with hard-negative candidate pools.
3. Add masked-edge pair reconstruction only if direct relation leakage can be removed in code; otherwise use source topology as a fixed factor only.
4. At inference, create parser-light response queries from exact surface span + whole parent context, with target value masked and non-target context retained.
5. Output top-k source owner beams plus soft relation-role scores.
6. Let typed comparators or finite verifiers decide support/conflict/U; then existing native code can run only on verified B/A subsets.

This is the smallest implementable graph model that addresses the generic ownership provider gap without reverting to free Qwen sourceQA or hallucination-label training. If it fails the heldout ownership tests, do not add more losses or a deeper GNN; fall back to typed schema contrast banks and report generic ownership as unresolved.
