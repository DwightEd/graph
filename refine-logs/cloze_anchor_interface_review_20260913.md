# Cloze anchor interface review after native v2 A-stage failure

Date: 2026-09-13
Reviewer mode: real collaboration-agent GPT-5.5 fallback. Codex MCP was not used. This is a focused `research-refine` interface design review, not a new proposal scoring loop and not an experiment-integrity verdict.

Scope: I read the frozen/current interface code paths `route_graph/atomic_anchor.py`, `route_graph/evidence_anchor.py`, `route_graph/audit_prompts.py`, the follow-up notes in `refine-logs/structured_alignment_followup_20260913.md`, a bounded snapshot of `outputs/native_audit_v2_20260913/A/*.json` and `outputs/native_audit_v2_20260913/reader_cache/*.json`, plus the not-yet-connected `route_graph/json_framing.py` prototype. I did not run GPU, did not alter the 19 frozen v2 implementation files, and did not treat the still-running v2 snapshot as final empirical results.

## Bottom-line review

The cloze revision is the right minimal next interface change. The current v2 failure is dominated by semantic anchoring and formatting at A, not by native graph evidence being absent. In the A files I read, the compiled atomic questions that reached self/source had 49 `invalid_question`, 11 `uncertain`, and 0 scored; the self failures were mostly wrong role/event answers or over-wide containing spans. The reader cache also contained 26 `invalid_json` frame outputs, 25 of which the prototype first-root parser would recover as a complete root object followed only by terminal punctuation. This is an interface bottleneck before B/C/D mechanism calls can be meaningfully interpreted.

The proposed change from a pure structured-role JSON question to a natural cloze over the original claim directly addresses the main failure mode: the current `question` is a JSON string such as `{"conditions": ..., "requested_role": "destination"}`, while `_check_question` asks the reader to answer from the full response using only that string. The checker often returns another role, the whole predicate phrase, or a surrounding span. Replacing only the target span in the claim with a unique `MASK` gives the reader a concrete surface slot to fill, so exact short-span recovery becomes a local alignment check rather than an underspecified semantic search.

This change is credible only if the scope is stated narrowly: it repairs the semantic anchor/contrast construction that gates native auditing. It does not prove that the frozen model internally used source information, and it must not be reported as evidence about native information absence if A still fails.

## Required blockers before connecting cloze to v3/native validation

1. **Separate source-visible and self-visible payloads.**

   The source QA payload must never contain the unmasked claim, `answer_quote`, self answer, target aliases, or any decontext field that reveals the target. It may contain the masked claim, requested role, non-target role table, exact previous-link metadata, and non-target decontext replacements. The self QA payload may contain the unmasked current claim, but only because it is checking that the mask recovers the selected response slot. That self result must be marked as a slot-local recoverability check, not a truth check.

2. **Make the cloze object first-class, not only a string.**

   The current `evidence_anchor._check_question` passes only `response` and `question` to SELF_QA, and `_source_anchor` passes only the question string to source QA. If conditions move out of the natural question string, the reader and validators need a stable sidecar object. Minimum schema:

   ```json
   {
     "id": "stable question id",
     "claim_quote": "original response claim, self-only",
     "answer_quote": "target response span, self-only / audit-only",
     "masked_claim": "same claim with exactly one unique MASK",
     "mask_token": "[[MASK:<id>]]",
     "requested_role": "subject|predicate|object|...",
     "role_bindings": [{"role": "...", "surface_quote": "...", "surface_span": [0, 0], "referent_quote": null}],
     "condition_roles": [{"role": "...", "surface_quote": "...", "referent_quote": null}],
     "previous_link": {"relation_to_previous": "...", "previous_claim_quote": "..."},
     "hidden_answer_aliases": ["..."]
   }
   ```

   Keeping `question` as a human-readable cloze string is fine for IDs and logs, but source/verify binding should use the structured sidecar. A map from `question_id` to compiled metadata inside `AtomicReader` is an acceptable minimal implementation; relying on the cloze string alone is too brittle.

3. **Rework condition binding around role ids/spans.**

   The proposed condition table should not be concatenated into the cloze string, because that duplicates evidence-like text and can reintroduce target leakage. But then the old validation `quote_span(question, condition_quote)` is no longer the right invariant. The validator should require one `condition_check` per non-target `condition_role`, with the `condition_quote` equal to that role's surface or decontext value from the sidecar and source citations exact in the source. It should not require the condition quote to be found inside the natural `question` string.

4. **Run the hidden-alias leak guard over every source-visible field.**

   The current compiler already checks hidden target aliases against conditions and premise quotes. Cloze makes this more important. The guard must cover `masked_claim`, condition rows, decontext referents, previous-link fields, source prompt payload, and any generated natural question copy. It should allow the answer only inside `claim_quote` for SELF_QA, where that field is explicitly self-only. If a non-target decontext replacement shares the same referent span as the target or overlaps a target alias, reject that masked role while preserving the rest of the frame.

5. **Do not let SELF_QA become circular semantic validation.**

   SELF_QA can answer from the current claim because its purpose is to verify that the compiler selected a recoverable slot and preserved the rest of the claim. It cannot validate source grounding. The strongest implementation is deterministic first: replacing `answer_span` by `MASK` and then restoring `answer_quote` must exactly recover `claim_quote`; the reader then checks only `faithful_to_current_claim`, `condition_roles_preserved`, and `ambiguous_within_claim=false`. Any answer drawn from another sentence in the full response should invalidate or downgrade the question.

6. **Keep multi-role source mismatch as unknown.**

   The follow-up SRLScore/FENICE notes are right: same-event role structure helps, but nearest-event or highest-average role similarity is not evidence. If the source candidate disagrees on several roles, or a non-target role cannot be grounded in the same event as the target, the label must remain `U/unknown`. E/C require all non-target conditions supported in the same source event; N requires all non-target conditions supported and only the target unstated. Missing or conflicting premises cannot become N.

7. **Parser recovery must be auditable and counted separately.**

   The `json_framing.py` prototype is acceptable in spirit for pure boundary noise: it parses one complete root object, rejects incomplete roots, second objects, prose tails, duplicate keys, arrays, and non-finite numbers, and records the discarded suffix plus raw hash. Two required adjustments before use:

   - Markdown-fenced JSON is currently stripped and may be reported as `strict`; it must either be rejected or counted as a recovered formatting case with the stripped prefix/suffix recorded.
   - The run summary must report `strict_json`, `framed_recovered`, and `invalid_json_unrecovered` separately. Recovered examples should keep the exact raw output and suffix. This prevents quiet conversion of format failures into clean reader success.

## Assessment of the four proposed changes

1. **Natural cloze over the original claim: accept, with source/self payload separation.**

   This is the smallest change that actually attacks the observed self-answer mismatch. It should use the original claim surface with exactly one target span replaced by a unique mask. For source QA, use the masked/decontextualized question copy only; do not send the original target or unmasked claim.

2. **SELF_QA explicitly scoped to current claim: accept, but narrow its claim.**

   Giving SELF_QA the current claim should reduce wrong-role and whole-span answers. It should return a local slot answer and ambiguity flags. It must not be described as independent evidence or as a source-grounding verifier, because it necessarily sees the response assertion.

3. **First-root JSON parser/generation stopper: accept for boundary errors only.**

   The observed raw cache supports this: most invalid JSON examples were complete objects followed by a stray `]`, `}`, or `%`. This parser should not repair missing fields, complete strings, choose among multiple objects, or ignore content-bearing tails. It is a framing normalizer, not a semantic retry.

4. **Unknown on multi-role source error: required.**

   This is not optional conservatism. It is needed to avoid turning a wrong event into evidence for or against a single slot. If all roles around an event drift, the system has no identified contrast and should report anchor failure, not pick the most similar source event.

## Specific fraud / false-positive risks and minimal fixes

- **Target leakage through decontextualization.** If the target is a pronoun or shares a referent with another role, replacing only the surface target may still expose the target through a referent field. Fix: compute aliases from surface target, target referent, and all overlapping referent spans; reject only the leaking target role.

- **Cloze grammar can leak weak type information.** Number, gender, tense, and article choice can make some answers easier. This is acceptable as a response-slot check, but sourceQA should still require exact event evidence and unresolved-candidate handling. Do not call this a blind truth label.

- **Condition table no longer lives in the question string.** The current validator expects quoted conditions to be located in `question`; this will falsely invalidate good cloze questions or encourage duplicating the table into text. Fix: bind condition checks by role id and sidecar value, not by searching the natural question.

- **SELF_QA can answer from the wider response context.** The prompt should say the answer must fill the mask in `current_claim`, while full response context is only for antecedents and qualifier preservation. Return a flag or citation span proving the answer was taken from the current claim.

- **Parser truncation can become cherry-picking.** A first-root parser can hide bad generations if it drops a second object or explanatory text. Fix: reject any tail containing a second JSON start, letters/digits, quotes, or prose; record exact suffix for every recovered item; keep strict and recovered denominators separate.

- **Derived/paraphrastic source answers can collapse exact-span logic.** If the source supports the masked claim only by derivation and has no exact answer quote, E/C slot replacement should not proceed unless the downstream contrast can be expressed with a verified exact replacement. Otherwise leave the semantic status uncertain and do not enter native mechanism certification for that question.

## Minimal interface revision that I would adopt

Define an `atomic-cloze-question@1` artifact and make it the only object that can enter source/verify/compare:

1. Compiler creates role spans as before, then produces `masked_claim` by replacing one exact target span with `[[MASK:<id>]]`. It asserts exactly one mask and no target alias in any source-visible field.
2. SELF_QA receives `response`, `current_claim`, `masked_claim`, `question_id`, requested role, and condition roles. It must return the exact response span filling the mask, whether all non-target conditions remain represented, and whether another span in the current claim could also fill the mask. This step only validates the compiled question.
3. SOURCE_QA and VERIFY receive `source`, `masked_claim`, requested role, non-target role table, and previous-link metadata. They do not receive the unmasked claim, answer quote, self answer, target aliases, or native scores. They must map every non-target row to exact source quotes and answer the mask only if the same event is identified.
4. COMPARE uses the frozen response `answer_quote`, source answer/verifier outputs, and condition-table validity. It keeps the current rule that invalid E/C/N mass moves to U.
5. The anchor report exposes denominators: extracted frames, rejected frames, rejected masked roles, strict JSON parses, framed JSON recoveries, invalid unrecovered JSON, self-local failures, source-condition failures, unresolved multi-candidate cases, and scored anchors.

## Decision

Proceed with the cloze anchor interface as the next minimal revision, but only after fixing the seven blockers above. This is an interface repair for A-stage anchoring. It should not be used to claim native mechanism absence or internal information insufficiency unless it produces valid E/C/N anchors and the later B/C/D mechanism certificates succeed under the frozen protocol.
