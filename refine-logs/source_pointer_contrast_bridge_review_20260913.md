# source pointer contrast bridge review — 2026-09-13

Scope: bounded design review only. I did not modify the frozen `soft_graph_v1` code or start GPU work. I inspected the current proposal anchor, native contrast/operator interfaces, pairedR/MLP certificate code, and the proposed bridge. I also checked the cited Mixing Mechanisms paper enough to verify the relevant constraint: §3.4 separates a reflexive pointer signal from an answer payload by using a counterfactual answer absent from the original input. For this project, a target-token effect is not binding evidence, and even a raw-input -> selected V-message -> recipient F origin test only identifies an input source and mediated path unless additional address/pointer-specific evidence separates owner/role addressing from answer payload flow.

## Verdict

Adopt the bridge, but only as a strict certificate subroutine for a bounded subset:

> Frozen source pointer/literal candidate + frozen response role pointer -> deterministic single-slot edited event -> finite validation of original unsupported and edited supported -> existing full-event CausalOracle / pairedR / MLP checks.

This is enough to produce real comparable B/A events in natural data for scalar, entity, date/time, quantity, location, duration, and attribute slots when a source candidate is already fixed and the edit is deterministic. It is not enough for whole-event fabrication, multi-role errors, unsupported source absence, predicate/condition rewrites, or ambiguous owner repair. It should therefore attach to the strict A/B certificate layer, while the soft graph detector remains the full-coverage prediction path.

The main improvement over v1-v3 is structural: the alternative value is no longer a free-text source-QA answer. It comes from a frozen source node ID and a deterministic surface policy. The reader only performs finite validation over two fixed statements and an exact highlighted source occurrence. That reduces the open QA failure mode without pretending the reader is ground truth.

## One implementable plan

### Inputs already available

For each soft-graph source term selected before C/D:

- source node ID, source kind, source span, exact source quote, source token keys;
- literal decoded value and `value_type` when the source graph is a literal-field graph;
- field path, event/record membership, and same-schema control metadata;
- response target role ID, role name, exact role span/quote, response event anchor span/quote;
- non-target role spans, condition spans, and extraction kind;
- matcher marginal `pi`, pool status, and local finite relation scores.

No source answer string is generated. No RAGTruth label enters this construction.

### Deterministic surface policy

Build exactly one candidate edited event `A` from the observed event `B` by replacing the target response role span only. Reject instead of normalizing creatively.

Allowed first-pass replacements:

1. **Pointer-role source node**: replacement surface is the exact source substring of the fixed source node.
2. **Literal string**: replacement surface is the decoded scalar string, without Python quotes, only if it is a nonempty printable scalar and the target role is not itself a quoted/code literal. v1 should reject quoted/code target slots entirely, because quote delimiters, escaping, and literal-vs-natural-language interpretation otherwise require a second rewrite policy.
3. **Literal int/float**: replacement surface is the exact raw literal text. Preserve response-side unit text only if that unit lies outside the target role span and finite validation says the edited event is source-supported with matching unit/scope.
4. **Literal bool**: reject in v1. Do not verbalize `True`/`False` into “open”, “available”, “closed”, negation, or any task-specific lexical enum inside this bridge.
5. **Literal None**: reject for support contrast. `None` may explain unknown/withholding, but it cannot produce `editedclaim S>=0.8` and must not become a “correct answer”.

Reject ranges, normalized times, currency/unit conversions, and paraphrased values unless the exact source surface can be inserted as a grammatical target role and then passes finite validation. `hours-range` cases therefore pass only when the source scalar surface itself is a valid replacement in the response event; no hidden range-to-duration converter should be added in this bridge.

### Role restrictions

Start v1 with these target roles only: object, quantity, time, location, duration, attribute, origin, destination. Reject subject, predicate, negation, condition, quoted/code slot, and bool replacement in v1. The subject restriction is reasonable: even a contiguous NP can control agreement, possessives, reflexives, appositives, and later references inside the event, so allowing it would turn a single-slot bridge into a hidden rewrite system.

The edit script must prove that every character outside the target role span is identical between `B` and `A`. If the replacement makes the event ungrammatical, creates duplicate incompatible mentions, or leaves old coreference behind, the bridge returns `invalid_single_slot_edit`.

### Finite C validation

Use one finite verifier stage after the candidate edit is frozen. It may see the complete source, the original event B, the edited event A, the target role metadata, the exact highlighted source node, and non-target role/condition spans. It must not see labels, native deltas, or later certificate outcomes.

Required finite decisions:

- `q_original` over S/C/N/U for the original event B; require `C+N >= 0.8`.
- `q_edited` over S/C/N/U for edited event A; require `S >= 0.8`.
- `q_source_node_relation` over supports/contradicts/unrelated/unknown for the **fixed highlighted source node with its actual owner/event/record**; require supports for A.
- `q_preservation` over pass/fail/unknown for owner, non-target roles, conditions, tense/scope, and grammar; require pass.

The exact-source-node relation is the key anti-circularity gate. If A is supported somewhere in the full source but not by the selected frozen source node/record/event, the contrast may be useful as a semantic repair candidate but cannot be used for origin or pairedR/MLP source certificates. Mark it `support_elsewhere_not_selected_node`.

All four decisions remain model predictions. They condition the mechanism audit; they are not ground truth. Final factual accuracy must still be evaluated against independent labels after predictions/certificates are frozen.

### Contrast object

For each passing case, emit:

- `contrast_id`, source node ID, response role ID, event ID;
- B event text and A event text;
- target span in B and A, source span, source keys, response token span;
- deterministic edit script and rejection checks;
- tokenizer input sequences, common prefix length, branch token lists, slot masks;
- finite verifier distributions and raw predictions;
- `scope="conditional_single_slot_source_pointer_contrast"`;
- `not_ground_truth=true`;
- `pointer_payload_distinction="answer_surface_replaced; origin_mediation_identifies_input_path_not_abstract_binding_pointer"`.

Then pass only this frozen contrast to the existing CausalOracle. Use the existing F definition: `F = log P(B complete event | shared prefix) - log P(A complete event | shared prefix)`. Do not length-normalize. Do not collapse to first differing token except as a diagnostic.

## Edge cases and required failure statuses

### Multiple source occurrences

A repeated value is allowed only if the source node ID and membership are exact. The finite verifier must affirm that the selected node’s owner/event/record supports edited A. If the same value appears under multiple records and C cannot bind the selected occurrence, return `ambiguous_same_value_origin`. Source reuse across response mentions is allowed; no one-to-one capacity constraint should be added.

### Multiple wrong roles

If replacing one role creates a hybrid event that is still unsupported, return `multi_role_or_whole_event_required`. Do not search a second replacement in this bridge. This prevents a same-event multi-role mismatch from being turned into a fake single-slot repair.

### Missing / N cases

For original N where no applicable source node exists, this bridge cannot build a supported A. Do not revive the old withholding template here. If a candidate source node exists and single-slot replacement makes A supported, it is a valid `unsupported_value_vs_source_supported_value` contrast even if the original label was N rather than C. If no selected source node supports A, the strict certificate layer abstains; the soft detector may still output an N_soft prediction.

### Units and decoded literals

Do not infer units from field names as if they were source text. Field path may help the verifier decide support, but replacement text must be from source quote/decoded scalar plus unchanged response context. Unit mismatch, currency conversion, time normalization, or range-to-duration conversion returns `unit_or_normalization_required`.

### Subject, bool, and quoted-slot replacement

Subject, bool, and quoted/code target slots are intentionally closed in v1. Return `subject_coreference_unsafe`, `bool_enum_rewrite_required`, or `quoted_slot_rewrite_required` rather than trying to preserve meaning through hidden grammatical or task-specific rewrites. This conservative boundary is appropriate for the first implementation because the bridge is meant to test deterministic single-slot contrasts, not to introduce another generator.

### Pointer vs payload

A passing B/A contrast shows that the model’s probability preference over two fixed continuations changes under a graph operation. It does not by itself show that the model stored a source pointer rather than an answer payload. The raw-input -> selected V-message -> recipient F mediated origin test is necessary for source-origin claims, but it is still not sufficient for abstract binding identification: the selected input’s answer payload may simply flow through V. To claim a pointer/address mechanism, the audit would need additional evidence that owner/role/address information is retained while answer content is separated, for example interventions that preserve the selected owner/event address while changing content, or content swaps that leave address selection unchanged. This distinction follows the same concern as Mixing Mechanisms §3.4: an intervention that changes the answer token can conflate the dereferenced payload with the pointer used to retrieve it.

## Reuse of pairedR and MLP interactions

The old pairedR/MLP operators can be reused without becoming toy demonstrations, but only after a bridge contrast passes.

For pairedR, the valid direction is:

- source position/origin effect opposes original B: `delta_source <= -0.5` under the same F;
- history effect supports original B: `delta_history >= 0.5`;
- pairedR moves fixed history attention mass to the validated source destination and compares against same-mass unrelated source destinations.

This becomes a meaningful “history over source” witness because F now compares an observed unsupported event against a validated single-slot source-supported event. It is still a transport/path witness, not proof that the transported object is an abstract binding pointer. If the bridge only has target-dependence without A/B, pairedR should not run.

For MLP interaction, reuse the existing 2x2 design only when the validated source evidence opposes B and the MLP branch supports B under the same F. Slot/context decomposition remains necessary; if context dominates or the MLP scope is broader than the evidence group, keep the existing `broad_MLP_event_interaction` downgrade.

Neither operator proves the unique cause of the hallucination. They only certify a conditional mechanism path for a fixed contrast.

## What this bridge does not solve

It does not produce a full detector score. It does not cover arbitrary N, whole-event hallucination, multi-role repair, source extraction failure, or cases where correct support requires paraphrase, arithmetic, normalization, or a generated rewrite. It does not eliminate Qwen: C still supplies finite relation/preservation predictions. It only removes the open answer-generation interface from strict contrast construction.

The design should therefore report two denominators:

1. soft detector denominator: all claims/roles/words scored by the soft graph path;
2. strict bridge denominator: candidates where deterministic single-slot source-pointer contrast was attempted, passed, and then received mechanism certificates.

Failure to pass the second denominator is not evidence that the model lacks internal source/history dependence. It only means this strict contrast bridge could not construct a valid comparable event.

## Minimal acceptance criteria for implementation

Before connecting this to a new run, require CPU tests for these interface cases:

- literal `None` cannot create a supported A;
- same value in two source records is rejected unless the selected record is verified, and quoted/code target slots are rejected in v1;
- numeric scalar replacement preserves or rejects units explicitly;
- one wrong role plus another wrong role fails edited-S validation and returns `multi_role_or_whole_event_required`;
- subject replacement is rejected in v1;
- exact non-target character preservation is checked before CausalOracle;
- C verifier can only validate the frozen highlighted source node, not another source occurrence;
- contrast object tokenizes into non-overlapping B/A continuation events and records common prefix/slot masks.

If these hold, this is a reasonable next bridge to test after `soft_graph_v1`, with the claim restricted to conditional strict certificates for source-pointer-derived single-slot contrasts. It should not be marketed as binding-pointer identification unless a later address/content separation test is added and passes.

## Source note

Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context, arXiv:2510.06182v2, submitted 2025-10-07 and revised 2026-05-28. I used the abstract and §§3.2-3.4 from arXiv HTML to ground the pointer-vs-answer-payload warning.
