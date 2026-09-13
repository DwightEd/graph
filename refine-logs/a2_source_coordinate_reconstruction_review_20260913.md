# A2 deterministic source-coordinate reconstruction review — 2026-09-13

Scope: bounded `research-refine` method review after stopping Qwen source-template synthesis. I did not edit code, browse literature, read labels, or run GPU. This review only judges whether the proposed deterministic source-coordinate reconstruction is a viable next data formulation for the single `GroundedGraphAdapter` line.

## Verdict

Proceed only if the claim is narrowed to **source-coordinate conditional reconstruction pretraining plus natural detection evaluation**. This is a better A2 bridge than Qwen-generated templates because the owner coordinate and copied payload are now mechanically true. It removes the observed semantic weak-label failures such as dangling accepted text and arbitrary invented targets.

It still does not solve generic natural ownership by construction. Training on copied source blocks under the original prompt can teach copying, source formatting, and local coordinate recovery. It does not teach that a natural answer phrase means “overall rating,” “business name,” “all seven days,” or a QA answer unless the frozen observer state and graph adapter transfer that signal. That transfer must be treated as the experiment’s question, not as an assumption.

## Focused viable formulation

Use one dataset schema per selected source block:

```json
{
  "source_id": "...",
  "task": "QA|Summary|Data2txt",
  "source_sha256": "...",
  "split": "source_train|source_validation",
  "prompt_prefix": "the unchanged original prompt through the source/question",
  "continuation_text": "exact copied source block or field fragment",
  "continuation_source_map": [{"char_span_in_continuation": [a,b], "source_node_ids": ["..."]}],
  "pointer_targets": [{"token_index": t, "positive_node_ids": ["..."], "target_kind": "first_token_of_copied_field|first_token_of_context|bundle_block"}],
  "ce_tokens": "all continuation tokens",
  "semantic_ground_truth": false,
  "hallucination_labels_used": false
}
```

Data construction should be deterministic:

- Split by raw source SHA before any example construction; use the same 240-source plan only as the first bounded run.
- For QA/Summary, use exact source text-context blocks with complete raw spans and exact source-coordinate maps. Do not ask Qwen to restate or choose relations.
- For Data2txt, use exact field payloads inside their actual record-local raw context. Avoid value-only continuations, because the pre-value hidden state would lack the property/owner cue and would mostly train a surface copier.
- Use all original source nodes in the graph. Pointer positives must be coordinates that are mechanically copied from the source: field/context/component owners, or a bundle node only when the supervised target is the copied bundle block, not a single value pretending to express an all-days semantic claim.
- Apply pointer loss only at the first token of a copied owner span or other predeclared coordinate anchor. Full-token CE may cover the whole continuation, but later subword/value tokens should be reported separately because teacher forcing exposes earlier pieces.

The loss can stay minimal:

```text
L_gen = mean CE over continuation tokens
L_ptr = mean -log sum_{n in positives(t)} A_tn over mechanically mapped pointer tokens
L = L_gen + λ L_ptr
```

Keep the existing single adapter, graph/no-edge variants, and frozen LM baseline. Do not add relation heads, hallucination negatives, Qwen semantic filters, or synthetic wrong examples at this stage.

## Required countercontrol against copy/format shortcuts

Add exactly one source-side ablation before interpreting any training win:

**Owner-payload erasure control.** For source-heldout reconstruction evaluation, rerun feature capture/inference with the copied owner payload span replaced by a typed sentinel while preserving the rest of the prompt, record context, graph node identity, node type, and non-target text. Measure value-token and pointer-token deltas:

```text
copy_dependence = NLL_erased(value/pointer tokens) - NLL_full(value/pointer tokens)
graph_gain = NLL_noedge - NLL_graph on the same full source
```

A model that only learns answer formatting or the language prior should retain most non-value CE gains under erasure and should not preserve coordinate accuracy on hard same-surface cases. A model using source content should lose value-token support and pointer confidence when the true payload is erased. A graph-specific claim also needs `graph_gain` over the no-edge adapter on hard cases such as same-field other-record values, homographs, address/rating numeric collisions, and seven-day bundles.

This control uses no hallucination labels and no artificial wrong examples. It is a deterministic source-view intervention on the weak source-coordinate task.

## Failure bounds to pre-register

- **QA question mismatch remains unresolved.** If the original prompt asks a question but the continuation is a copied source paragraph, the adapter may learn “copy relevant source-looking text,” not answer ownership. Natural RAGTruth full-word/post-first evaluation is the first meaningful test of transfer.
- **Data2txt formatting remains a shortcut.** Raw Python-dict fragments expose keys, quotes, commas, and value positions. Report CE separately for field keys/punctuation, non-value context, and first value tokens; only value/pointer-token gains matter for grounding.
- **Bundle supervision is not semantic quantifier proof.** A copied seven-day source block can train a bundle coordinate. It cannot prove that a later natural answer’s “every day” requires all seven days unless a separate natural evaluation or typed verifier supports that interpretation.
- **Pointer accuracy is not truth.** Even a correct source coordinate distribution is only an applicability proposal. It does not certify S/C/N/U, B/A correctness, or native adoption.
- **Distribution shift can make the run negative.** If graph/no-edge/LMNLL show no natural detection gain, that is a real negative result for this bridge, not evidence that the internal model has no information.

## Minimal natural evaluation

After freezing the adapter and controls, evaluate on the same natural full-word and strict post-first splits already used by the project:

1. frozen LM NLL,
2. A2 no-edge adapter score,
3. A2 graph adapter score,
4. A2 graph with owner-payload erasure only for source-heldout reconstruction diagnostics, not natural label fitting.

Predefine the signed score direction from the model spec before reading labels. Report all words, all unavailable words, and post-first spans. Do not stop at source-heldout reconstruction accuracy, and do not call graph attention a causal route. Native B/A work remains conditional on later exact contrast construction.

## Bottom line

This deterministic A2 plan is the smallest coherent replacement for the failed Qwen-template data branch. It gives the adapter real source-coordinate targets without hallucination labels and keeps the single joint-model story intact. The launch should be blocked only if the implementation cannot provide exact continuation-to-source maps, first-token pointer targets, source-heldout split isolation, graph/no-edge/LMNLL baselines, and the owner-payload erasure control. Everything else should wait for the frozen natural evaluation instead of becoming another prompt-format loop.
