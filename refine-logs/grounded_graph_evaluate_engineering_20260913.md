# Grounded graph evaluator engineering review

Scope: `next_iteration/grounded_graph_evaluate.py` only. This is a static/CPU
integrity review. No RAGTruth annotation file was read, no prediction/evaluation
run was launched, and this establishes no detection result.

## What is already correct

The evaluator requires a complete natural-prediction manifest and its exact
entry roster before joining labels. It requires a JSON prediction for every
prepared natural response and an NPZ only for available entries, then verifies
prediction code, feature manifest/code, and artifact bytes. It retains
unavailable words in every subset with their explicit status instead of
dropping them.

The fixed prediction protocol supplies scores directly; `ranking` does not
fit, normalize, or flip any score after labels. It uses inverse source word
counts as sample weights, which gives each source equal total mass. The masks
are correct: through-first-error includes the first error token, while
post-first-error begins strictly after it. Task/generator/official-split and
training-source-overlap groups are all emitted.

Two synthetic CPU tests cover source-balanced weighted AUROC/AUPRC without a
direction flip, and all-word/first-inclusive/strict-post-first denominator
accounting with unavailable words. They pass; Ruff is clean.

## Critical: 0. Required: 3

1. **Freeze the evaluator before any annotation-byte access.** `run` currently
   hashes `response.jsonl` before writing the audit settings and executed
   evaluator snapshot. Move evaluator/settings snapshot publication before
   even the label-byte hash; only then hash and parse labels. Prediction
   manifests are already checked first. This preserves the stated
   label-after-frozen-code boundary.

2. **Rebuild and require the complete word roster.** The evaluator checks only
   word-span bounds. It must require prediction words, in order, to equal the
   exact `re.finditer(r"\S+", annotation["response"])` spans and text. Otherwise a
   self-consistent prediction record can omit or overlap words and reduce the
   all-word/post-first denominator. Keep unavailable words in that exact roster.

3. **Bind the canonical completed population before using its label location.**
   The supplied population directory is currently used only to read a settings
   file containing a dataset label hash. Require the completed canonical
   population marker, bound population settings/input manifest and label-free
   state, record their hashes in the audit, and reject a modified or incomplete
   parent before label access. Also validate every annotation label span and
   its stored text against the response, and explicitly reject a missing label
   for any frozen prediction entry. The full annotation file may validly have
   additional IDs beyond the natural prediction roster, so do not require a
   false bidirectional ID-set equality.

If these are fixed, this evaluator can support the predeclared all-word and
post-first **error-ranking** comparison. It still cannot establish source owner
truth, semantic support, native routing, or a causal effect.

## R1–R3 closure recheck

All three requirements are now closed. The evaluator freezes audit settings and
an executed copy of itself before its first annotation-byte hash or parse. It
requires the prediction word `(span, text)` sequence to equal the exact
non-whitespace response roster, so unavailable words stay in the same complete
denominator. It validates canonical population completion/progress/full
annotation-free roster and records hashes of the five parent files in the
audit; each frozen prediction row is then matched by digest to that roster.

The new evaluator protocol requires every annotation span to supply
`start/end/text` and verifies its text against the response slice. This is a
new strict engineering contract for this evaluator, not a claim about the
scope of earlier population evaluators. It additionally requires the canonical
population settings to declare `labels_used: false`. Extra annotation IDs
outside the frozen natural roster remain valid; any missing prediction ID is
rejected.

The two synthetic CPU ranking/mask tests still pass and Ruff is clean. The
evaluator is **0 Critical / 0 Required** for a later label join. No real label
file was read and no evaluation result exists. Any future result remains an
error-ranking result with predeclared direction, not an owner, semantic, or
native-route result.
