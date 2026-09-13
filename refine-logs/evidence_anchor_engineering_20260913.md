# Evidence-anchor semantic interface engineering review — 2026-09-13

Reviewed only `route_graph/frozen_reader.py`, `route_graph/audit_prompts.py`,
and `route_graph/evidence_anchor.py` against the semantic A/C interface in
`refine-logs/FINAL_PROPOSAL.md`. This was a two-pass review, including an
independent leakage/provenance pass and CPU regression checks. `causal_groups`
and any end-to-end detector or GPU result are outside this review.

## Result

**Do not merge until the Critical and Required findings are fixed.** The prompt
wording is generally careful about treating payload text as data and keeping
the response answer out of `SOURCE_QA`; the defects below are in the executable
interface and output contract.

### Critical

1. **The N/NULL “independent” source verification is primed with the first
   source reader's full conclusion** — `route_graph/evidence_anchor.py:128-134,
   171-176`. `_source_anchor()` passes the complete `SOURCE_QA` object as
   `source_answer` to `VERIFY_SOURCE`: it includes `answerability`, proposed
   answer, citations, coverage flags, and unresolved candidates. A first-pass
   `not_stated` result is therefore disclosed to the supposedly independent
   verifier, whose agreement is then sufficient for an N score. This violates
   the proposal's two independently obtained not-stated estimates
   (`FINAL_PROPOSAL.md:67,72`) and can convert a correlated/rubber-stamped NULL
   into a high-risk anchor. Run a second blind source-plus-question answerability
   call for N; do not provide the prior answerability, evidence set, or status.
   A separate citation/condition check may receive a proposed source answer for
   C/E, but it must not count as the independent N confirmation.

### Required

1. **Citations are only unlocated substrings, not source occurrences mapped to
   tokens** — `route_graph/evidence_anchor.py:43-49, 121-142`. `_quotes_valid`
   accepts any matching string and deliberately allows repetitions; no output
   records character spans, occurrence IDs, or token mappings. Thus an evidence
   quote such as `duration 7` can be credited to the wrong stage when that text
   appears twice. The proposal requires exact source substrings mapped to tokens
   and ownership/condition checks (`FINAL_PROPOSAL.md:60-63`). Require cited
   spans or occurrence IDs from the reader, validate them against the full
   source, map them with the tokenizer, and persist that mapping. If a string
   citation is ambiguous, retain all resolved occurrences only when each is
   semantically valid, otherwise abstain.

2. **Source QA omits task context** — `route_graph/evidence_anchor.py:23-27,
   121-134`. The source interface accepts only source and question, while the
   specified source QA uses full D, task, and question (`FINAL_PROPOSAL.md:59`).
   A question may not repeat task-level constraints. Thread `row['task']`
   through both blind source calls and make it part of the cached payload.

3. **A grounded contrast can replace the response with an unrelated source
   substring** — `route_graph/evidence_anchor.py:185-229`. For C, the code only
   checks whether `source_answer_quote` occurs somewhere in the source. It does
   not require it to be the proposed source answer or a validated answer
   citation. A scripted CPU case with `source_answer='8'`, evidence `['8']`, and
   `source_answer_quote='7'` produces an `available` edit that replaces `9`
   with `7`. Require an explicit, validated link between the replacement and
   the supported source-answer/citation occurrence, otherwise return
   `contrast_requires_rewrite`. Store the full edited response and an explicit
   absolute character/token edit script once it is built; local claim-only text
   cannot yet establish the required complete B/A events with the same
   termination boundary (`FINAL_PROPOSAL.md:87-90`).

4. **The reader cache does not enforce deterministic frozen execution or a
   complete execution identity** — `route_graph/frozen_reader.py:34-64`.
   `ask()` accepts a training-mode model, so dropout can make the first cached
   answer non-deterministic. The cache key also relies on opaque caller-supplied
   `model_identity` and omits a reader/parser revision, tokenizer/chat-template
   identity, and rendered-prompt hash; cached parsed predictions survive a
   parser change even though `raw_output` is present. Require eval mode, accept
   or derive a validated composite execution identity (model/tokenizer/template/
   generation/parser version), and include the rendered prompt identity in the
   immutable request. On a concurrent cache publication, load and verify the
   already-linked request rather than leaking a `FileExistsError`.

5. **Coverage does not classify nonassertions or preserve their exact spans** —
   `route_graph/evidence_anchor.py:232-334`. `EXTRACT` asks for
   `nonassertion_quotes`, but `build_anchors()` never validates, resolves, or
   uses them. Such words become generic `score=.5, abstain=true`, indistinguishable
   from assertion coverage failure. The specification requires every word to be
   assertion-covered, explicitly nonassertional, or `coverage_unknown`
   (`FINAL_PROPOSAL.md:52-53,65,74`). Resolve nonassertion spans with the same
   exact/ambiguity rules, emit a per-word coverage state, and keep unresolvable
   text explicitly `coverage_unknown`.

### Optional

1. The answer-leak guard is lexical (`evidence_anchor.py:60-65`): a paraphrase
   or presupposition of the answer can pass unless the fallible self-check
   catches it. Add adversarial paraphrase tests and treat an uncertain self-check
   as `invalid_question`; retain the self-check record.

2. Validate input-row schema and cached-file schema before field access. This
   would turn malformed `source_span`, stale cache JSON, and missing response
   hashes into controlled abstentions/errors rather than incidental exceptions.

## Confirmed behavior

- The extract call contains task and response but not source; `SOURCE_QA`
  itself receives only source and question, excluding the response answer and
  risk score. The prompts explicitly describe payload fields as quoted data.
- E/C are gated on answerable/supported condition coverage, citations, premises,
  and no unresolved candidates. N is kept as a reader estimate rather than a
  world-absence claim, but the Critical independence defect currently prevents
  its intended gate from being valid.
- Invalid reader JSON, context exhaustion, malformed question spans, and
  invalid source citation strings become failure/uncertain records instead of
  being silently repaired. The current word loop preserves all response words
  with score `0.5`/abstain when no valid question covers them.

## Verification

Added [test_evidence_anchor_review.py](/share/home/tm902089733300000/a903202310/lys/research/graph/tests/test_evidence_anchor_review.py), a CPU-only regression file.
It presently fails in the two expected places:

```text
test_source_verification_does_not_receive_prior_answerability_label
test_grounded_replacement_must_be_an_evidence_citation
```

The requested interpreter ran those tests: `2 failed in 4.47s`. Ruff passed:

```text
/tmp/research_lint_20260912/bin/ruff check route_graph/frozen_reader.py route_graph/audit_prompts.py route_graph/evidence_anchor.py tests/test_evidence_anchor_review.py
All checks passed!
```

Manual checks also confirmed a repeated source citation is accepted without an
occurrence mapping and that `source_qa_request()` has only `source` and
`question` keys. No GPU work or implementation changes were made.

## Re-review after semantic-interface and alignment revisions

The first-round findings are retained above as history. This section evaluates
the revised three semantic modules and `audit_alignment.py`; a narrow read of
`causal_groups.build_contrast()` was used only to verify the B/A hand-off, not
to review its oracle or search implementation.

**Result: Critical 0, Required 3.** The blind NULL fix, task propagation,
ambiguous-citation abstention, word coverage states, frozen-reader eval/cache
identity improvements, full B/A construction hand-off, and branch-ID alignment
are materially fixed.

### Required

1. **A claimed source-answer quote is not itself grounded to a source span** —
   `route_graph/evidence_anchor.py:121-149, 167-179`. Evidence quotes are
   resolved or ambiguity downgrades the verification to uncertain, but neither
   reader's non-null `source_answer_quote` is passed through `quote_span()` or
   tied to an evidence span. Two blind calls can therefore agree on an invented
   answer quote while citing an unrelated valid source sentence; `complete`
   permits E/C before `_make_edits()` later rejects the replacement. Resolve
   and persist answer-quote spans, require both exact quotes to be valid and
   tied to the verified evidence set, and gate E/C on that result. The added
   `test_source_answer_quote_must_resolve_to_the_source` currently fails on
   this case.

2. **Condition ownership remains a bare reader boolean instead of an auditable
   citation coverage table** — `route_graph/audit_prompts.py:47-50, 64-67` and
   `route_graph/evidence_anchor.py:167-178`. The revised spans identify whole
   cited strings, but the output still has only `all_conditions_covered` and
   `all_premises_supported`. It cannot show which evidence span supports the
   entity, action, stage/time, negation, quantity, and unit, or whether a joint
   multi-sentence set covers them. The A/C interface requires that table
   (`FINAL_PROPOSAL.md:60-63`). Request condition-to-citation-span IDs, validate
   them against the resolved evidence spans, and gate E/C on complete coverage.

3. **The cache key still omits mutable generation configuration** —
   `route_graph/frozen_reader.py:68-80, 114-120`. It now includes validated
   model/tokenizer/code identity, rendered prompt hash, parser revision, dtype,
   torch version, eval checks, and a safe concurrent-publication comparison.
   `model.generate()` nonetheless consults mutable `model.generation_config`
   such as EOS/forced-token/beam settings that are neither explicitly supplied
   nor keyed. A changed generation config can reuse a stale cached prediction.
   Include a canonical generation-config hash in the request or explicitly pass
   and key every generation-affecting setting.

### Resolved findings

- `VERIFY_SOURCE` now receives only source, task, question, and
  `source_answer=None`; it does not see the first pass's answerability,
  citations, or answer. N remains gated on two independently obtained
  `not_stated` outcomes (`evidence_anchor.py:121-185`). The original Critical
  NULL leak is resolved.
- Source requests now carry task. Repeated evidence strings are resolved to
  character spans when unique; ambiguity moves the effective verification to
  uncertain and records the citation error. `audit_alignment.span_keys()` is
  the exact source-character-to-frozen-token mechanism for the persisted spans.
- Nonassertion spans are resolved with the same ambiguity rule and every word
  now exposes `coverage_state` as assertion, nonassertion, or coverage_unknown.
- `FrozenReader` rejects training mode, keys rendered prompt/parser/dtype/torch
  details, validates composite identity fields, and handles a link-publication
  race by comparing the complete record.
- The B/A event itself is now constructed in the hand-off
  `causal_groups.build_contrast()`: it uses the original response through the
  same sentence boundary, applies only the absolute answer span replacement,
  validates the original against frozen replay IDs, and delegates duplicate/
  prefix rejection to `ContinuationContrast`. This resolves the prior concern
  that `_make_edits()` alone was only claim-local.
- `audit_alignment.slot_masks()` now re-tokenizes each complete branch and
  compares IDs with `prefix[prompt_length:] + continuation` before slot/context
  accounting. The new mismatch regression passes.

### Re-review verification

```text
python -m pytest -q tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py
3 passed, 1 failed

/tmp/research_lint_20260912/bin/ruff check route_graph/frozen_reader.py route_graph/audit_prompts.py route_graph/evidence_anchor.py route_graph/audit_alignment.py tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py
All checks passed!
```

The one expected failure is the ungrounded `source_answer_quote` regression
above. No GPU work or implementation changes were made in this re-review.

## Final re-review: answer grounding, condition tables, and generation identity

**Result: Critical 0, Required 1.** The remaining Required item is a Ruff
standards failure; the three semantic/cache findings from the preceding section
are resolved.

### Required

1. **Ruff rejects the condition-table type error** —
   `route_graph/evidence_anchor.py:154-156`. A non-list
   `condition_checks` is a type violation but raises `ValueError`; Ruff rule
   `TRY004` requires `TypeError`. Change that exception class, then rerun the
   recorded lint command. No behavioral or model change is needed.

### Resolved findings

- Each non-null `source_answer_quote` is uniquely resolved in the full source,
  assigned `source_answer_span`, and required to occur within cited evidence.
  A missing/ambiguous quote downgrades verification to uncertain. The previously
  failing fabricated-quote regression now passes.
- Both source prompts request `condition_checks`, and the backend validates
  condition quotes against the question, cited source quotes against the full
  source, status vocabulary, and nonempty citations for supported conditions.
  E/C additionally require nonempty valid tables whose entries are all
  supported for both independent readers.
- `FrozenReader` now snapshots `model.generation_config`, applies fixed greedy
  settings, records the complete configuration in its cache request, and passes
  an explicit `GenerationConfig` to `generate`. This closes the stale-cache
  path caused by mutable generation settings.
- `audit_alignment.slot_masks()` retains the complete B/A event-ID equality
  check introduced in the prior revision.

### Final verification

```text
python -m pytest -q tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py
4 passed in 9.46s

/tmp/research_lint_20260912/bin/ruff check route_graph/frozen_reader.py route_graph/audit_prompts.py route_graph/evidence_anchor.py route_graph/audit_alignment.py tests/test_evidence_anchor_review.py tests/test_audit_alignment_review.py
1 error: TRY004 at route_graph/evidence_anchor.py:156
```

No GPU work or implementation changes were made in this final re-review.

## Closure re-check: lint correction

**Result: Critical 0, Required 0.**

`route_graph/evidence_anchor.py` now raises `TypeError` for a non-list
condition table, satisfying Ruff `TRY004`. The previously recorded scope was
rechecked with:

```text
/tmp/research_lint_20260912/bin/ruff check route_graph/frozen_reader.py route_graph/audit_prompts.py route_graph/evidence_anchor.py route_graph/audit_alignment.py
All checks passed!
```

The prior semantic findings remain resolved: blind second-source verification,
unique cited answer grounding, per-condition source spans, full B/A token
identity, and frozen generation/cache identity. No GPU work or implementation
changes were made in this closure check.
