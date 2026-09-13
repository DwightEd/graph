# Natural semantic bridge review after SourceRel failure

2026-09-13. Bounded `research-refine` design review. I read the current SourceRel failure notes, `automatic_query_scoring_refinement_20260913.md`, and the complete inventory witness. I also checked the primary InPars and Promptagator method descriptions because the proposed Route A explicitly invokes that style. No code edits, GPU, downloads, or label reads.

## Bottom-line decision

Adopt **Route B as the next mainline**: a frozen Qwen3 reasoning-enabled graph-pointer semantic bridge that jointly selects evidence and predicts support/conflict/not-stated/unknown for fixed response claims. Treat every output as a model prediction, not ground truth. Use it to produce complete detector outputs and, when possible, deterministic B/A contrasts for downstream native localization.

Do **not** make Route A the next mainline. LLM-generated natural masked queries with known source IDs are useful for training a candidate retriever, and InPars/Promptagator justify the idea that LLM-generated in-domain retrieval queries can help. But A still returns owner candidates under weak source-generated positives. It does not by itself decide applicability, conflict, not-stated, multi-role constraints, or B/A validity. After the SourceRel negative result, another candidate-only run is the wrong next bottleneck.

The previous review over-constrained the problem when it implied that using a frozen semantic classifier at inference is an oracle leak. It is not. A classifier/verifier prediction is allowed as part of the method. The leak occurs only if its outputs are called GT, used to tune on the evaluation labels, or used to certify mechanism success without independent evaluation.

## Why raw late-interaction alone is insufficient

The complete inventory fixed a real representation bug: all sources, long text fields, unknown-valued owners, components, and bundles are now addressable. It did not create an automatic semantic provider. A raw untrained Llama/Qwen token MaxSim over target-masked text can help retrieve related spans, but the actual failures require relation meaning:

- Santa Barbara/California should map to location/address evidence rather than business name.
- Numeric address components should not be forced into rating pools.
- Overall business rating must be separated from review-star rows.
- All-days hours require a response-side quantifier/condition decision, not just seven source keys.

These are entailment/applicability decisions. MaxSim can supply candidates and receipts; it cannot decide the relation. A complete method must output a factual prediction even when no native certificate exists. Otherwise the detector remains a high-precision auditor with near-zero coverage.

## Route A review: source-generated natural pointer training

### What A can plausibly fix

A better version of SourceRel can train on naturalized source-to-query pairs instead of source JSON reconstruction. For each source inventory node or bundle, a frozen LLM generates masked natural claims that require that source owner. The known source occurrence or bundle ID becomes a weak positive. This is analogous to InPars, where an LLM generates queries from documents and high-probability query/document pairs train a ranker, and Promptagator, where task-specific prompts and consistency filtering create synthetic retrieval data for a target task.

A minimal A interface would be:

```json
{
  "synthetic_query_id": "...",
  "source_id": "...",
  "masked_natural_claim": "The business is located in <VALUE>.",
  "target_slot_type": "text|number|time|date|unknown",
  "positive_owner_ids": ["source bundle/member IDs"],
  "candidate_pool_ids": ["same-source inventory IDs"],
  "weak_label_origin": "source_generated_not_hallucination_GT",
  "heldout_source_split": "source_train|source_validation|official_test_source"
}
```

Train only owner retrieval:

```text
score(q, G_s) = query-conditioned cross-attention over source graph nodes/bundles
L_owner = -log Σ_{p∈P(q)} exp score(q,p) / Σ_{c∈pool(q)} exp score(q,c)
```

Use hard negatives that match the observed failures: same source same value type, same field other record, same value different owner, same owner different relation, review-star versus business-star, weekly day endpoint versus whole schedule, and unknown-valued attribute owners. The model output remains an owner beam with graph receipts.

### Why A should not be next mainline

A's labels are generated from the source, not from natural responses. They are weak positives for a retrieval task. They can improve query style transfer, but they cannot settle whether a real response claim is supported, contradicted, absent, or conditionally applicable. The generator can also leak field semantics into the synthetic style, overproduce clean declaratives that differ from real RAGTruth errors, and teach a retriever to solve the generator's distribution rather than the response distribution.

Cross-attention over the full source graph is stronger than the previous single-vector head, but it also adds training cost and design degrees of freedom before the semantic interface is fixed. If A is run now, the likely output is another report saying “candidate recall changed, truth still unknown.” That is exactly the cul-de-sac we should avoid.

A is worth doing later only as an amortization or recall layer if B shows that graph-pointer semantic predictions are useful but expensive or source prompts exceed context. In that role, A supplies top-k candidates to B; B remains the semantic decision point.

## Route B review: frozen graph-pointer semantic bridge

### Why B is the smallest adequate architecture now

B directly addresses the missing semantic provider while preserving the research boundary. It uses a frozen model as an inference-time predictor, not as GT. It can produce a full detection score and a pointer-level evidence hypothesis for every claim. It also can condition exact B/A construction for the subset where a repair or withholding contrast is explicit.

This is the minimal complete pipeline:

```text
complete source inventory
  -> fixed response claim/slot surfaces
  -> graph-pointer semantic bridge (Qwen3 reasoning-enabled)
  -> prediction output: S/C/N/U + evidence IDs + applicability + contrast if possible
  -> detector score for all claims
  -> native localization only for frozen verified-contrast subset
```

No SourceRel training is required before this. No first-token multi-gate is used. No reader output is called truth.

### Input contract

For each response claim target or parent claim, freeze a packet before the Qwen call:

```json
{
  "packet_id": "response_id + claim_id + source_inventory_sha",
  "task": "Data2txt|QA|Summary",
  "response": {
    "full_text": "...",
    "prior_context": "all text before the parent",
    "parent_text": "complete readable parent/base unit",
    "target_slots": [
      {"slot_id": "...", "span": [a,b], "surface": "...", "surface_type": "..."}
    ],
    "target_masked_parent": "only selected target span(s) replaced by typed masks"
  },
  "source": {
    "full_source_available": true,
    "rendered_source_with_ids": "source text/fields with stable node IDs",
    "candidate_inventory": [
      {"id": "...", "kind": "field|component|bundle|context", "raw_span": [x,y], "quote": "...", "field_path": [], "value_status": "observed|unknown", "bundle_members": []}
    ],
    "inventory_denominator": {"nodes": 0, "omitted": 0, "mapping_unavailable": 0}
  },
  "allowed_output_labels": ["S", "C", "N", "U"],
  "label_meanings": {
    "S": "source supports the claim/target under stated conditions",
    "C": "source contradicts the claim/target under stated conditions",
    "N": "model predicts not stated in the given source; not a certified absence proof",
    "U": "uncertain/ambiguous/inapplicable/unavailable"
  }
}
```

If the full source fits context, include it with inline stable node IDs. If not, include source context envelopes plus all candidate nodes and record a `source_rendering_truncated_or_windowed` status. Retrieval prefiltering may be used only as a recall heuristic with a denominator receipt; it cannot silently define absence.

### Output contract

The bridge must emit a single structured prediction per packet:

```json
{
  "packet_id": "...",
  "prediction_scope": "claim|target_slot|multi_slot_parent",
  "status": "predicted|parse_failed|source_too_long|insufficient_inventory",
  "factual_prediction": "S|C|N|U",
  "confidence": 0.0,
  "applicability": "applicable|wrong_owner|ambiguous_owner|condition_mismatch|not_in_source|uncertain",
  "evidence_groups": [
    {
      "group_id": "...",
      "role": "owner|relation|condition|payload|counterevidence|absence_search_region",
      "source_node_ids": ["..."],
      "source_raw_spans": [[0, 0]],
      "quoted_text": "...",
      "pointer_validated_against_inventory": true
    }
  ],
  "history_links": [
    {"previous_claim_id": "...", "relation": "reuse|correction|quote|new_topic|uncertain", "prediction": "..."}
  ],
  "contrast": {
    "kind": "repair|withhold|none",
    "original_text_B": "exact original parent or suffix event",
    "edited_text_A": "exact repair or withholding text",
    "edited_spans": [[0, 0]],
    "source_payload_node_ids": ["..."],
    "non_target_preservation": "predicted_preserved|not_preserved|uncertain",
    "eligible_for_native": false
  },
  "not_ground_truth": true,
  "semantic_certificate_count": 0
}
```

Pointer validity is programmatic: every cited ID exists, every cited quote matches the raw source span, and every edited span matches the original response text. Semantic correctness is not programmatic; it is the model's prediction.

### Detector score

For a detector, B can produce a prediction for every packet. A simple frozen score is enough:

```text
risk = P(C) + P(N) + 0.5 * P(U)
```

If the model only emits one label plus confidence, map it conservatively:

```text
S: risk = 1 - confidence
C/N: risk = confidence
U: risk = 0.5, confidence_for_ranking low
```

Do not let native effects flip a claim to supported. Native effects can add localization and confidence receipts only when the semantic relation is known or predicted with a frozen output. Unknown native dependence remains localization, not truth.

### Native localization handoff

For `C` or high-confidence `N` with an explicit contrast:

1. Freeze the bridge packet and prediction before native work.
2. Construct B/A from exact response spans and source payload IDs. If the edit is not single-span safe or preservation is uncertain, output a prediction with no native certificate.
3. Define source groups from `evidence_groups[*].source_node_ids`, not from post-hoc native effects.
4. Define history groups only from explicit `history_links` that are not correction/quote/new-topic.
5. Run the existing CausalOracle/validate-origin/paired-role/MLP checks with matched controls selected before seeing native effects.
6. Report native output as conditional: “for the bridge-predicted contrast, these source/history messages affected F.” Do not call it original-generator causality or internal-only discovery.

This answers the automatic lookback/localization requirement better than candidate-only runs: every prediction has a cited source/historical hypothesis; the subset with exact B/A gets a mechanism certificate.

## What is GT and what is prediction

No semantic GT is required to **run** the method. Qwen predictions, synthetic source queries, graph retrieval scores, and native effects are all model outputs.

Semantic GT is required only for claims about accuracy:

- factual detection performance needs independent RAGTruth or human labels;
- span localization quality needs independent span labels or human review;
- evidence-owner quality needs independent owner/bundle annotations if claimed directly;
- “graph improves detection over Qwen” needs the same frozen evaluation labels for graph and no-graph baselines.

Semantic GT is not required for these narrower claims:

- the pipeline produced complete prediction artifacts for N claims;
- every cited source pointer is a valid raw-span pointer;
- a deterministic edit preserved non-target text programmatically;
- a native intervention changed the frozen observer's B/A log-odds under a frozen contrast;
- source inventory coverage improved over the prior short-scalar inventory.

The paper must keep these apart. The bridge is a frozen semantic predictor, not an oracle. Its errors are method errors and must count in final evaluation.

## Honest evaluability under the existing split

The old 36-response roster is development-exposed. It can diagnose failures and document artifacts, but it should not be the final evidence for accuracy. The existing source split matters differently for A and B:

- For B, there is no trained component, so source-train membership is not a training leak. The prompt/schema has been shaped by old36 failures, so freeze it before evaluating on official test rows. Report source-seen/source-heldout anyway because source familiarity may affect any auxiliary learned retrieval component used later.
- For A, train only from official train source inventories, select hyperparameters on source-heldout validation inside train, and reserve official test sources for evaluation. A's synthetic owner labels are weak retrieval labels; they cannot become natural-owner GT.

Minimal evaluation blocks:

1. **Execution/integrity:** full denominator, parse failures, pointer validity, source rendering/windowing, inventory coverage, and no label reads before prediction.
2. **Detection:** compare bridge predictions against independent RAGTruth labels on frozen official test/population splits. Baselines: Qwen no graph pointers/full source, Qwen with graph pointers but no native, previous soft/candidate systems, and simple lexical/late-interaction retrieval if included.
3. **Localization/certificate:** among bridge-predicted contrast-eligible cases, report B/A construction validity, native effect/control/sham pass rates, source/history/MLP certificate counts, and the denominator of ineligible cases. Do not hide abstentions.

No arbitrary pass threshold is needed before running the method. Use resource gates only: if parse or source rendering breaks, fix interface; if predictions run but accuracy is poor, report a negative result.

## Recommended implementation order

1. Freeze the graph-pointer bridge schema and rendering on the existing complete inventory.
2. Run a CPU prompt assembly/parser/pointer-validity preflight on old36 without labels.
3. Run Qwen3 reasoning-enabled prediction on old36 and a small official-test dry slice only to catch interface failures, not to tune thresholds.
4. Once interface is stable, run full frozen prediction before reading labels.
5. Evaluate against independent labels and compare to no-graph Qwen. This decides whether graph pointers actually help detection.
6. Only then launch native localization for contrast-eligible predictions.
7. Consider Route A only if B is semantically useful but too slow or context-limited; train A as a candidate prefilter/distillation target, not as the semantic authority.

## References checked

- InPars: Bonifacio et al., 2022, uses LLM-generated document queries as synthetic positive query/document pairs and filters generated pairs by sequence probability before training retrieval/reranking models. Primary source: https://arxiv.org/abs/2202.05144 and HTML mirror https://ar5iv.labs.arxiv.org/html/2202.05144.
- Promptagator: Dai et al., 2022, uses few-shot task-specific LLM query generation, generated-data consistency filtering, and retriever/reranker training. Primary source: https://arxiv.org/abs/2209.11755.

These papers support LLM-generated weak retrieval data as a practical training source. They do not make weak generated owner labels into factual truth labels, and they do not replace the semantic bridge needed for support/conflict/not-stated prediction.
