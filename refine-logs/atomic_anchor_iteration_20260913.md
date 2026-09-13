# Atomic anchor revision motivated by natural-v1 failures

Status: experimental compiler implemented but NOT imported by the running v1.
No additional GPU experiment has been launched. V1 must finish, preserve its
immutable results, and restore the population before integration and v2 execution.

## Observed failure, no annotation join

At 26/36 completed A responses: 110 proposed questions, 74 invalid, 26 uncertain,
10 supported, zero unsupported/native-selected questions. 50 self-check answer
quote mismatches, 17 hidden-answer/premise overlaps, and 9 source-reader invalid
JSON events were recorded. These counts overlap and are not independent errors.
Raw examples show paragraph summaries used as answer slots, and SELF_QA returning
the full explanation instead of the intended answer constituent. Some source QA
outputs put an entire evidence sentence into source_answer_quote while returning
a shorter source_answer. These are front-end identification/interface failures.
They cannot support a conclusion that native internal information is insufficient.

The frozen full36 run is allowed to finish. No thresholds or questions are being
changed mid-run. Full word and invalid-window denominators remain visible.

## Primary methods read for this revision

FActScore, Appendix A.2 and official atomic_facts.py: decompose individual sentences
into atomic statements; the released biography implementation uses fixed and
retrieved demonstrations and retains sentence/paragraph groupings. This motivates
explicit unit addressing, but its biography-specific filtering and fact-count
precision are not adopted as our error-span evaluation.
Sources: https://aclanthology.org/2023.emnlp-main.741.pdf ;
https://github.com/shmsw25/FActScore/blob/main/factscore/atomic_facts.py

DnDScore, sections 3–4 and prompt discussion: adding disambiguating context can
turn an atomic target into a multi-fact assertion. Their joint decomposition and
decontextualization returns paired forms, and verification distinguishes the
subclaim from its added context. This motivates keeping target, surface binding,
and antecedent binding separately typed rather than mixing them into a summary.
Source: https://arxiv.org/html/2412.13175v1

D-FActScore, sections 4–5: independently supported facts can be misleading when
they are attributed to different entities but narrated as one individual. The
method groups facts, links an entity to each group, then verifies against that
entity's evidence. Our corresponding design requirement is that all conditions
must belong to the same event; matching separate role values across unrelated
source events is insufficient.
Source: https://aclanthology.org/2024.findings-acl.160.pdf

OpenFActScore, sections 3–4: adapting open models requires the actual chat template
and explicit generation/verification prompts; its comparison separates fact
generation from fact verification. It does not establish that our local Qwen3-8B
reader is accurate. Its model/table scores are not reused as our measurements.
Source: https://arxiv.org/html/2507.05965v1

These sources were read directly. No bibliographic-helper verification or new
novelty approval is claimed.

## Implemented prototype

route_graph/atomic_anchor.py compiles extracted role frames into answer-hidden
questions. Each frame contains one predicate, a subject, and explicit arguments
or qualifiers, each grounded to an exact response span. Python independently
masks every role while retaining the other role values in structured conditions.
Four text units are requested per extraction call; the full response remains
visible for interpretation, so this batching does not discard distant context.

Surface and antecedent bindings remain separate. Antecedents must precede the
current clause. Hidden-answer aliases and shared antecedent spans are checked
against all conditions/premises; leaking masks are recorded as rejected targets,
while other legal masks remain available. A role-balanced fixed cap selects
questions without any truth labels. Existing SELF_QA, two blind source readings,
semantic class guards, and native certificates still run; none is replaced by
compiler success.

The prototype preserves previous-clause relation proposals for later independent
slot-level linkage. No paraphrased claim is substituted into the original replay.
Minimal-answer prompts distinguish the value quote from the containing evidence.

Five targeted CPU tests pass; independent scoped review closed its two Required
items. This validates parsing/binding invariants, not extraction accuracy.
The prototype is not yet connected to audit_prepare or included in the v1 frozen
code manifest. Integration must occur only after the active v1 is finished.

## Next checks and limitations

- Bind every non-target role to its own source condition-check entry; a generic
  all_conditions_covered boolean or one whole-question citation is insufficient.
- Preserve invalid JSON as observed failure; inspect exact syntax separately
  from semantic error. Do not call a model to repair answers silently.
- Test the same full36 roster with a new versioned run, and report changes to
  window availability, all-word coverage, actual native calls, and error recall.
- Distinguish lexical/antecedent alias checks from complete semantic leakage
  detection. The latter still depends on an imperfect independent reader.
- Exact single-slot edits and high-precision source anchoring may still omit
  multi-error facts, derived answers, and non-quotable alternatives. Do not
  rename those omissions as successful mechanism localization.


## Integration update

Atomic version2 now integrated in audit_prepare; frozen code identity includes compiler and evaluator metrics dependency. Exhaustive condition-role coverage implemented for both blind source passes; missing non-target premises cannot yield target-absence N. Protocol drives8 batches/48 frame-proposals per response including invalid proposals, with unprocessed unit IDs explicitly retained; final question cap48. Two independent integration reviews closed Required;25 relatedCPU tests passed. Same36 development roster prepares into native_audit_v2_20260913. No new factual/effectiveness result yet. V1 result and independent integrity audit are negative/not-supported, archived separately.
