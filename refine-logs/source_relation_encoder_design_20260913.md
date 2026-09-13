# Source-relation encoder design review — 2026-09-13

Scope: bounded `research-refine` method review for a source-only self-supervised graph relation encoder. This review uses the local notes `fact_scope_primary_methods_20260913.md`, `semantic_provider_decision_20260913.md`, `source_self_supervised_owner_review_20260913.md`, and `unsupervised_detector_lit_20260913.md`. I did not browse new literature, start GPU work, edit implementation, or read labels.

## Decision

Adopt the idea only if it is narrowed to a **source relation ownership encoder**. This is the smallest trainable structure that can plausibly address the generic provider bottleneck left by typed-hours: it learns to bind a masked role/constraint context to the right source occurrence or source event under source-derived supervision, with no hallucination labels and no free-text Qwen source answer.

Do not call it a hallucination detector, support/conflict verifier, internal attribution module, or native certificate generator. Its first valid claim is: **given a response event/role query, propose relation-consistent source owner candidates and NULL/UNKNOWN under a source graph view, while resisting same-value/different-owner and same-field/different-condition shortcuts**. Truth direction still needs deterministic typed comparators, finite relation checks, or later independent evaluation.

This route is stronger than another prompt/format repair because the target is identifiable from source coordinates. It is also stronger than the earlier single owner head if it includes pair/hyperedge relation reconstruction. Single masked-value retrieval alone will mostly learn lexical/value similarity and field priors.

## Problem anchor check

The user wants automatic lookback, constraint ownership, hallucination span/range localization, and propagation analysis under a graph method, preferably without training on hallucination labels. Typed-hours can produce nonzero native contrasts for one narrow Data2txt family, but it does not solve generic QA/Summary ownership or unknown constraints. A source-only relation encoder preserves the no-hallucination-label anchor while adding the missing general candidate-supply mechanism.

The scope must stay honest:

- Source-derived training can supervise **ownership under the source view**.
- It cannot supervise **whether a generated response is true**, except where source/value comparators or later verifiers provide that relation.
- It cannot prove internal-only sufficiency or original-generator causality.
- NULL in this model means `unmatched_under_candidate_sourceview`, not global not-stated or hallucinated.

## Minimal architecture: SourceRel

### 1. Source graph contract

Every training item starts from a sealed source graph, not from hallucination labels. The graph should support both structured Data2txt and natural text, but the typed semantics are only as strong as the source graph construction.

Nodes:

- `record` / `document` / `sentence_or_clause` / `event`;
- `role_value` nodes for entities, scalar values, time spans, locations, names, list items, booleans, NULL literals, quoted spans;
- `field_path` nodes for structured sources;
- `condition` nodes for time, stage, owner, negation, quantifier, source-local qualifiers;
- optional `shared_variable` nodes when the source graph explicitly links repeated mentions or fields to the same entity/event.

Edges:

- `contains`, `field_child`, `event_role`, `condition_of`, `same_record`, `same_event`, `same_variable`, `list_member`, `review_item`, `temporal_scope`, `negation_scope`, `next_clause`;
- no edge may be created from response labels, Qwen E/C/N, native effects, RAGTruth spans, or generated B/A outcomes.

For Data2txt, the source graph can come from AST/literal field paths plus typed adapters. For natural sources, the initial graph should use sentence/clause envelopes and conservative surface spans; if an LLM extractor supplies event roles, those roles must be frozen and evaluated as a noisy parser, not as truth labels.

### 2. Training examples

A training example is `(source_id, event_or_record_id, role_or_field_id)`.

Construct a masked query view by removing the target role/value and all target aliases from the event/record context:

```text
query = serialize(event context with target role <MASK_ROLE>, owner, predicate/field, non-target roles, conditions, negation, quantifier)
```

The candidate set is all source occurrences in the same source/document or a large same-source pool:

```text
C(q) = observed source nodes with compatible broad type + hard negatives + NULL + UNKNOWN
```

Positive set `P(q)`:

- the exact source occurrence(s) bound to the masked role in the same event/record;
- repeated mentions only if the source graph explicitly proves they are the same event/field fact;
- multi-positive list values only when the source graph marks the role as a coordinated multi-value slot.

Same value in another event is a hard negative by default, not a positive. If source identity is underspecified, the example is `ambiguous_source_identity` and excluded from ordinary positive CE.

### 3. Leakage controls

The query view must remove:

- exact target surface, normalized target value, aliases, range endpoints, units when the unit uniquely identifies the hidden target, boolean polarity words tied to the target, quote variants, and Data2txt display strings;
- absolute source char/token position;
- candidate IDs, rank, native effect, B/A status, Qwen labels, gold labels.

The candidate view may contain the source surface and provenance because the task is retrieval, but the scorer cannot use absolute co-location between the original unmasked occurrence and the masked query. Use canonical local serialization or randomized source order for training. Position-shuffle validation is mandatory.

For response inference, the response target surface must also be masked before ownership scoring. Value comparison belongs to typed/finite relation verification, not to candidate selection, otherwise wrong values will drag the model to the wrong owner by lexical similarity.

## Model

Use one trainable component: a role-aware relation scorer with optional shallow source-graph message passing. Do not start with a general GNN stack.

Representations:

```text
q_r = QueryEncoder(masked event/field context, role type, value type, condition flags)
c_j = CandidateEncoder(source occurrence surface, parent event/record context, field/path tokens, value type)
g_j = SourceGraphContext(candidate node after 0-2 typed message-passing layers)
```

The frozen text encoder may be the existing observer or a separate encoder, but it must encode the masked query view and candidate view, not the unmasked original target position. If a learned encoder is used, train it only on source reconstruction objectives below.

Scoring:

```text
s_r(q, c) = (P_r q)^T (Q_r g_c) / tau
          + a^T type_features(q, c)
          + b^T provenance_features(q, c)
          + role_bias_r
```

`type_features` may include broad role/value compatibility and normalized scalar family. `provenance_features` may include field-path tokens, source node kind, parent record/event kind, condition attachment count, and source-local graph distances. It must not include gold truth or native effects.

For multi-role event assignment at inference, add a small relation-coherence factor trained from source graph edges:

```text
psi(c_i, c_j, rho_ij) = score that two source candidates cohere under relation rho_ij
```

Then solve a reusable-source assignment, not a one-to-one matching:

```text
E(z | event) = sum_r -s_r(q_r, z_r)
             + sum_(r,r') -alpha * psi(z_r, z_r', rho_rr')
             + null_unknown_penalties
```

One source occurrence/event may be reused by multiple response roles or repeated response mentions. There is no source capacity constraint.

## Losses

### Unary owner retrieval

```text
L_owner(q) = -log [sum_{p in P(q)} exp s(q,p)]
                  / [sum_{c in C(q) ∪ {NULL, UNKNOWN}} exp s(q,c)]
```

This teaches source ownership, not truth.

### Pair/hyperedge relation reconstruction

Train relation coherence so the model distinguishes owner/context from surface similarity:

```text
L_pair = CE( psi(c_i, c_j, rho), y_same_relation )
```

Positives are source graph candidate pairs from the same event/record/condition relation. Negatives are same source, same value type, same or similar surface, but different owner/event/condition. For structured data, same field name in another record and same scalar under different field path must be explicit negatives. For natural text, same entity with different predicate/time/negation must be explicit negatives.

If this pair/hyperedge loss is absent, the model is not a relation encoder; it is just masked value retrieval and should not be expected to solve ownership.

### Deletion / NULL source-view objective

```text
L_null(q) = -log exp s(q,NULL)
                 / [sum_{c in C_deleted(q) ∪ {NULL, UNKNOWN}} exp s(q,c)]
```

`C_deleted` removes all positives and graph-proven duplicate positives while keeping a matched hard-negative pool. Candidate count, source coverage, type mix, and graph degree buckets must be matched so NULL does not become a missing-candidate-count detector. This label is only `unmatched_under_sourceview`, never not-stated truth.

### Shortcut penalties and ablations

Use hard-negative margin only against source-confusable negatives:

```text
L_margin = mean_n max(0, m + s(q,n) - logsumexp_p s(q,p))
```

Add entropy/temperature regularization only to prevent collapse to one field or NULL. Do not use it as a confidence calibration for factual correctness.

Total:

```text
L = L_owner + lambda_pair L_pair + lambda_null L_null + lambda_margin L_margin + lambda_reg L_reg
```

## Hard negatives required

A batch that lacks hard negatives cannot support an ownership claim. Required negative families:

- same source, same value type, different event/record;
- same surface/value, different owner/event;
- same field name/path in another record;
- same entity, different condition/stage/time;
- same predicate, different subject/object;
- same subject/value, opposite negation or boolean polarity;
- numeric same unit but different endpoint/range;
- NULL literal versus concrete value;
- high hidden-cosine candidate with incompatible source graph relation;
- nearby raw-token decoys that are not the source graph target.

## Inference interface

For each response event/role, output a candidate assignment object, not a truth label:

```json
{
  "schema": "source-relation-owner@1",
  "response_event_id": "...",
  "response_role_id": "...",
  "query_view_hash": "...",
  "target_surface_masked": true,
  "source_graph_hash": "...",
  "candidate_rankings": [
    {
      "source_node_id": "...|NULL|UNKNOWN",
      "source_event_or_record_id": "...",
      "rank": 1,
      "score": 0.0,
      "owner_probability_like": 0.0,
      "relation_coherence": 0.0,
      "hard_negative_flags": ["same_value_different_event"],
      "basis": ["masked_hidden", "field_path", "event_context", "condition_graph"]
    }
  ],
  "event_assignment_beam": [
    {
      "role_to_source": {"role_id": "source_node_id"},
      "energy": 0.0,
      "same_event_coherence": 0.0,
      "null_roles": [],
      "ambiguous_roles": []
    }
  ],
  "status": "candidate_supplied|ambiguous|null_under_sourceview|unknown_coverage",
  "downstream_allowed": "relation_verification_or_typed_comparator_only"
}
```

Downstream rules:

- If a typed comparator can compare response value against the selected source owner, it may produce support/conflict and B/A.
- If only ownership exists, the output is candidate supply or target-dependence input, not factual correctness.
- If relation verifier is unavailable or unknown, report `semantic_relation_unverified`.
- If source graph coverage is incomplete, report `unknown_coverage`, not N.

## Can it distinguish relation owner from similarity?

Yes, but only under a stricter target than “recover the masked value.” The identifiable training signal is the source graph relation: which value belongs to which owner/event/condition. The model can learn to reject similar but wrong nodes if the loss includes same-value/different-event, same-field/different-record, same-entity/different-condition, and high-cosine graph-incompatible negatives, and if pair/hyperedge relation reconstruction is part of the objective.

It cannot do this from ordinary contrastive retrieval alone. If the positive is just the masked value and negatives are random, lexical similarity and field-path priors solve the pretraining task while failing the user’s real constraint ownership problem. A graph encoder is justified only if the heldout tests show gains specifically on owner/condition-confusable negatives over a no-graph bi-encoder baseline.

## Heldout evaluation before integration

No hallucination labels are needed for the pretraining validation. Required source-derived metrics:

1. `source-heldout top-k owner recall`: source/document IDs disjoint between train and validation.
2. `same-value wrong-owner rate`: how often the model picks a same surface/value in the wrong event.
3. `same-field other-record error`: Data2txt field/path memorization check.
4. `same-entity different-condition error`: owner retained but time/stage/negation wrong.
5. `pair relation AUC/F1`: whether candidate pairs assigned to a response event cohere as a source event/record relation.
6. `deletion NULL FP/FN`: with matched pool size/type/degree buckets.
7. `position-shuffle robustness`: source order randomized or canonicalized; no large recall collapse.
8. `target-surface ablation`: response/hidden query target removed; no reliance on wrong-value lexical pull.
9. `schema-heldout`: some field paths or relation templates withheld when possible.
10. `downstream candidate supply`: fraction of frozen response roles that receive top-k candidates accepted by typed/finite relation checks, compared to rule matcher and Qwen-free baselines.

Only after these pass should the encoder feed the native auditor or soft graph detector. Natural RAGTruth labels may evaluate final detection, but must not select checkpoints, tune NULL cost, or choose hard-negative families.

## Identifiable failure bounds

This model should abstain or downgrade in the following cases:

- the source graph failed to extract the relevant event/field/condition;
- the response role/event segmentation is missing or loses owner/condition context;
- the response relation is a paraphrase not represented by source graph roles or typed comparators;
- multiple response roles are wrong and event identity cannot be recovered from non-target context;
- source has several same-value/same-field candidates with no distinguishing condition in the query;
- the selected source owner is applicable but the response value requires semantic contradiction, arithmetic, or unit conversion not covered by a comparator;
- NULL is produced only because the search/index omitted the true source node;
- frozen hidden states fail the position-shuffle or target-surface ablation checks;
- high source ownership does not imply the observer model used that source information, which still requires native target/origin tests.

These are not minor edge cases. They define the honest boundary between source ownership retrieval and factual detection.

## What not to add

Do not add a hallucination classifier, RAGTruth-span supervised head, synthetic binary hallucination labels, free-text Qwen source answer generation, or native-effect training signal to this pretraining stage. Do not add a deep GNN before proving the bilinear + pairwise relation encoder fails on source-heldout hard negatives. Do not turn NULL into N. Do not make one-to-one OT/FGW capacity constraints that force repeated source facts into NULL; source facts must be reusable.

## Minimal implementation order

1. Freeze a source graph inventory and canonical masked query serializer for Data2txt plus a small natural-source surface/event graph, with leakage checks.
2. Generate source-only train/validation instances with hard negatives and matched deletion pools.
3. Train the role-aware bilinear scorer plus pair relation factor. Keep 0-1 layer graph message passing optional and ablate it.
4. Run source-heldout/schema-heldout validation. Reject if same-value wrong-owner or position-shuffle failures remain high.
5. Use the encoder only to supply top-k source owner candidates for frozen response events/roles.
6. Let typed comparators or finite relation verification decide support/conflict/N/U where possible.
7. Feed only verified B/A subsets to native causal certificates; unverified ownership can feed target-dependence/localization but not correctness claims.

## Final review

This is a credible next architecture for the generic provider bottleneck if the paper’s claim is phrased as **self-supervised source relation ownership for candidate supply and graph-based detection inputs**. It is not sufficient as a standalone hallucination detector. The decisive fix relative to earlier failed branches is not “add a graph network”; it is the pair/hyperedge source relation objective plus hard-negative protocol that makes owner/condition errors observable without hallucination labels.

If the team cannot build reliable source graphs and leakage-proof masked query views, reject this route and stay with deterministic typed-schema contrast banks plus explicit unresolved denominators. Training a relation encoder on noisy or leaky source views would create a more sophisticated version of the same sourceQA failure: high-confidence but unidentifiable ownership.
