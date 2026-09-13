"""Candidate v3 prompts; not imported by the frozen native-v2 run."""

from route_graph.audit_prompts import PREFIX

SELF = (
    PREFIX
    + """
Recover the phrase hidden by <MISSING_SLOT> in the question. The payload supplies
current_assertion (the exact unmasked assertion) and the complete response only
for its context. Return the minimal exact phrase from CURRENT_ASSERTION that
occupies the missing slot, including its units/prepositions if they are missing.
Do not answer a different assertion from elsewhere in the response. A pronoun
resolved in the question may correspond to its surface mention in the assertion.
Return {"answer_quote":str|null,"question_faithful":bool,
"all_conditions_preserved":bool,"answer_leaked_in_question":bool}.
Check that the question preserves all non-target qualifications and references.
If multiple slot fillings within current_assertion fit, return null. Do not infer
truth, use the source, or add world knowledge. This is only mask recoverability
and question fidelity, not a factuality judgment. The requested phrase is not
supplied as a separate target answer and must actually occur in the assertion.
"""
)

SOURCE = (
    PREFIX
    + """
Fill <MISSING_SLOT> in the question using only the ENTIRE supplied source. The
original response answer and unmasked assertion are deliberately not supplied.
The conditions object enumerates ALL non-target conditions, including unparsed
context fragments. Its keys identify separate condition rows; do not rename,
combine, duplicate or omit them, even when their text values repeat.
Return exactly this structure:
{"answerability":"answerable|not_stated|conflicting|uncertain",
 "source_answer":str|null,"source_answer_quote":str|null,
 "evidence_quotes":[str],"all_conditions_covered":bool,
 "condition_checks":[{"condition_role":str,"condition_quote":str,
                       "source_quotes":[str],
                       "status":"supported|missing|conflicting|uncertain"}],
 "unresolved_candidates":[str]}.
condition_role must be an exact key of conditions. condition_quote must copy that
key's COMPLETE exact value. Include precisely one row for EVERY supplied key.
source_quotes are exact source citations; supported rows require citations.
All conditions and the answer must belong to the SAME entity and event under the
same time/stage, direction, location, negation, modality and qualifications.
Matching an entity or value in another event is not enough. Read the whole masked
assertion: isolated matching words across unrelated events are not joint support.
source_answer and source_answer_quote must BOTH copy the SAME MINIMAL phrase
filling the mask, not its containing sentence. Put whole supporting sentences only
in evidence_quotes. A derived/paraphrased answer has source_answer_quote null.
Use not_stated only when non-target premises are supported but this source does
not specify the target. Missing/contradictory premises or multiple unresolved
source-event candidates make the result uncertain/conflicting, not not_stated.
Return no invented alternative answer. Source absence is a reader estimate, not
proof of falsity in the world. List unresolved relevant candidates explicitly.
"""
)

VERIFY = (
    PREFIX
    + """
Independently fill <MISSING_SLOT> using the full source and the conditions table.
No prior source answer, original response answer, or unmasked assertion is given.
Return exactly:
{"status":"supported|not_stated|partial|conflicting|uncertain",
 "source_answer":str|null,"source_answer_quote":str|null,
 "all_premises_supported":bool,"all_conditions_covered":bool,
 "condition_checks":[{"condition_role":str,"condition_quote":str,
                       "source_quotes":[str],
                       "status":"supported|missing|conflicting|uncertain"}],
 "evidence_quotes":[str],"unresolved_candidates":[str]}.
Produce exactly one condition_checks entry per conditions key. condition_role is
that exact key; condition_quote copies its complete value. Never merge keys with
identical text or replace role IDs with guessed labels. Supported rows need exact
source_quotes. Check the complete masked assertion, including unparsed fragments.
Ground all rows jointly in the SAME event, entity, stage/time, direction, modality
and negation. Matching fragments from unrelated events are not joint support.
For supported, source_answer and source_answer_quote must copy the SAME minimal
exact source phrase filling the mask; full supporting sentences belong only in
evidence_quotes. For a derived answer, source_answer_quote is null.
Use not_stated only for absent target information after non-target premises are
supported; unresolved/missing/conflicting premises require uncertainty. Keep any
unresolved relevant alternatives explicit. A closest or most similar event is not
a verified answer. Do not borrow the meaning of a condition from a different event.
"""
)
