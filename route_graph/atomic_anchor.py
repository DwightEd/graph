"""Atomic event-role questions with explicit target and condition bindings."""

import json
import re

from route_graph import audit_prompts as prompts
from route_graph.audit_protocol import PROTOCOL
from route_graph.evidence_anchor import build_anchors, quote_span, text_units

ROLES = {
    "subject",
    "predicate",
    "object",
    "origin",
    "destination",
    "time",
    "location",
    "duration",
    "quantity",
    "condition",
    "negation",
    "attribute",
}

FRAME_PROMPT = (
    prompts.PREFIX
    + """
Extract ATOMIC event/relation frames from the specified text units, while reading
the complete response for pronouns and conditions. Do not summarize a paragraph.
Return {"frames":[{"unit_id":str,"anchor_quote":str,
"roles":[{"role":str,"quote":str,"referent_quote":str|null}],
"relation_to_previous":"same_claim|elaboration|correction|new_claim|unknown",
"previous_claim_quote":str|null}],
"nonassertion_unit_ids":[str]}.
Each frame expresses ONE predicate relation. anchor_quote is an exact contiguous
clause inside its unit. Each role quote is an exact minimal constituent inside
anchor_quote: subject, predicate, object, origin, destination, time, location,
duration, quantity, condition, negation, attribute. Include every explicit
argument, qualifier, unit and negation affecting this event. Split coordinated
relations into separate frames; keep their conditions. Roles must not overlap.
Use duration for a value together with its unit, e.g. "six weeks".
Use one role entry per role; combine a conjunction within one role if necessary.
Every frame needs subject and predicate and at least one other role.
predicate must be the minimal relational verb/phrase, not a whole sentence.
Do not put the entire assertion or a summary sentence inside a role.
For a pronoun or incomplete name, referent_quote may copy its unambiguous
antecedent from EARLIER response text; otherwise use null. It adds interpretation,
not source evidence or truth. Do not add world knowledge or infer missing facts.

Example, unit u1: "During loading, the crane moved the crate from the yard to the dock."
One frame has subject="the crane", predicate="moved", object="the crate",
origin="the yard", destination="the dock", condition="During loading".
These are roles in one event, not six unrelated facts. Copy only actual text.
For "The trial lasted six weeks." use subject="The trial", predicate="lasted",
duration="six weeks". Preserve any time/stage/negation written in the unit.
For an explicit continuation, elaboration, or correction, copy its exact previous
clause into previous_claim_quote; topic similarity alone is unknown. The caller
will independently verify the current-to-previous specific slot relation later.
Extract all distinct frames in the requested units within frame_limit.
"""
)

MINIMAL_SELF = (
    prompts.SELF_QA
    + """
The question is a structured event with exactly one requested_role absent from
conditions. Return ONLY the minimal verb or argument phrase filling that role.
Do not return its containing sentence, explanation, or the other conditions.
Do not add a determiner or remove units unless the response itself does so.
"""
)
MINIMAL_SOURCE = (
    prompts.SOURCE_QA
    + """
The question is a structured event with one requested_role absent from conditions.
source_answer and source_answer_quote must BOTH copy the SAME minimal phrase
filling that role. Full supporting sentences go only into evidence_quotes.
A source_answer_quote containing the whole evidence sentence is invalid unless
the requested role itself is that entire phrase. Preserve grammatical inflection.
Match all non-target roles to the SAME event, including stage, direction, negation.
For condition_checks include exactly one row per conditions key. Add condition_role
with that exact key; condition_quote must copy its complete individual value, not
the whole JSON. No extra roles and no combined rows. Evidence must belong to the
same event. A missing non-target role is an unresolved premise, not target absence.
"""
)
MINIMAL_VERIFY = (
    prompts.VERIFY_SOURCE
    + """
The question requests exactly one role in a structured event. Copy only its
minimal phrase into source_answer and source_answer_quote; put full citations
in evidence_quotes. Check each supplied non-target role against the SAME event.
For condition_checks include exactly one row per conditions key. Add condition_role
with that exact key; condition_quote must copy its complete individual value, not
the whole JSON. No extra roles and no combined rows. Evidence must belong to the
same event. A missing non-target role is an unresolved premise, not target absence.
"""
)


def response_units(response):
    return [
        {"id": f"u{index}", **unit} for index, unit in enumerate(text_units(response))
    ]


def compile_frame(
    response, frame, units, *, role_word_limit=16, predicate_word_limit=6
):
    """Bind roles to exact spans before constructing answer-hidden questions."""
    unit = units[frame["unit_id"]]
    anchor_span = quote_span(
        response, frame.get("anchor_quote"), (unit["start"], unit["end"])
    )
    raw_roles = frame.get("roles")
    if not isinstance(raw_roles, list) or not 3 <= len(raw_roles) <= len(ROLES):
        raise ValueError("frame requires three or more explicit nonoverlapping roles")
    roles = []
    seen = set()
    for item in raw_roles:
        if not isinstance(item, dict) or item.get("role") not in ROLES:
            raise ValueError("unknown atomic role")
        name = item["role"]
        if name in seen:
            raise ValueError("duplicate atomic role")
        seen.add(name)
        quote = item.get("quote")
        if not isinstance(quote, str):
            raise TypeError("role quote must be text")
        quote = quote.strip()
        span = quote_span(response, quote, anchor_span)
        if any(span[0] < r["span"][1] and span[1] > r["span"][0] for r in roles):
            raise ValueError("atomic roles overlap")
        if len(re.findall(r"\S+", quote)) > role_word_limit:
            raise ValueError("role is longer than the frozen constituent bound")
        referent = item.get("referent_quote")
        referent_span = None
        if referent is not None:
            referent_span = quote_span(response, referent, (0, anchor_span[0]))
        roles.append(
            {
                "role": name,
                "quote": quote,
                "span": span,
                "value": referent if referent_span is not None else quote,
                "referent_span": referent_span,
            }
        )
    if not {"subject", "predicate"}.issubset(seen):
        raise ValueError("frame lacks subject or predicate")
    if (
        len(next(r for r in roles if r["role"] == "predicate")["quote"].split())
        > predicate_word_limit
    ):
        raise ValueError("predicate is not an atomic relation")
    questions, rejected = [], []
    for target in roles:
        other = [role for role in roles if role is not target]
        # The hidden role's lexical value and its resolved referent are never
        # serialized into the question by the compiler.
        question = json.dumps(
            {
                "requested_role": target["role"],
                "conditions": {r["role"]: r["value"] for r in other},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        premise_quotes = list(
            dict.fromkeys(
                [r["quote"] for r in other]
                + [r["value"] for r in other if r["referent_span"] is not None]
            )
        )
        aliases = {target["quote"], target["value"]}
        if target["referent_span"] is not None:
            a, b = target["referent_span"]
            for role in other:
                span = role["referent_span"]
                if span is not None and span[0] < b and span[1] > a:
                    aliases.update((role["quote"], role["value"]))
        exposed = json.dumps(
            {
                "conditions": {r["role"]: r["value"] for r in other},
                "premise_quotes": premise_quotes,
            },
            ensure_ascii=False,
        )
        leaked = [
            alias
            for alias in sorted(aliases)
            if re.search(
                r"(?<!\w)" + re.escape(alias) + r"(?!\w)", exposed, re.IGNORECASE
            )
        ]
        if leaked:
            rejected.append(
                {
                    "target_role": target["role"],
                    "reason": "hidden_answer_alias_in_condition_or_premise",
                    "exposed_aliases": leaked,
                }
            )
            continue
        questions.append(
            {
                "claim_quote": frame["anchor_quote"],
                "answer_quote": target["quote"],
                "question": question,
                "premise_quotes": premise_quotes,
                "slot_type": (
                    "duration"
                    if target["role"] == "duration"
                    else "number"
                    if target["role"] == "quantity"
                    else "relation"
                    if target["role"] in {"predicate", "negation"}
                    else "entity"
                ),
                "predicate": next(
                    r["value"] for r in roles if r["role"] == "predicate"
                ),
                "relation_to_previous": frame.get("relation_to_previous", "unknown"),
                "previous_claim_quote": frame.get("previous_claim_quote"),
                "atomic_target_role": target["role"],
                "question_struct": json.loads(question),
                "hidden_answer_aliases": sorted(aliases),
                "atomic_role_bindings": [
                    {"role": r["role"], "quote": r["quote"], "span": r["span"]}
                    for r in roles
                ],
                "decontext_bindings": [
                    {
                        "role": r["role"],
                        "referent_quote": r["value"],
                        "referent_span": r["referent_span"],
                    }
                    for r in roles
                    if r["referent_span"] is not None
                ],
                "unit_id": unit["id"],
            }
        )
    return {"questions": questions, "rejected_targets": rejected}


def extract_atomic_questions(reader, response, question_limit=48, *, protocol=None):
    config = PROTOCOL if protocol is None else protocol
    batch_size = config["atomic_units_per_batch"]
    units = response_units(response)
    processed_units, considered_frames, omitted_frames = set(), 0, 0
    by_id = {u["id"]: u for u in units}
    questions, frames, failures, calls, nonassertions = [], [], [], [], []
    for start in range(0, len(units), batch_size):
        if (
            len(calls) >= config["atomic_batches_per_response"]
            or considered_frames >= config["atomic_frames_per_response"]
        ):
            break
        selected = units[start : start + batch_size]
        processed_units.update(u["id"] for u in selected)
        raw = reader.ask(
            FRAME_PROMPT,
            {
                "response": response,
                "units": [{"id": u["id"], "text": u["text"]} for u in selected],
                "frame_limit": config["atomic_frames_per_batch"],
            },
            max_new_tokens=config["atomic_frame_json_token_limit"],
        )
        calls.append(raw)
        proposed = raw.get("frames", [])
        if not isinstance(proposed, list):
            proposed = []
        permitted = {u["id"] for u in selected}
        available = min(
            config["atomic_frames_per_batch"],
            config["atomic_frames_per_response"] - considered_frames,
        )
        omitted_frames += max(0, len(proposed) - available)
        for frame in proposed[:available]:
            considered_frames += 1
            try:
                if not isinstance(frame, dict) or frame.get("unit_id") not in permitted:
                    raise ValueError("frame does not belong to the frozen unit batch")
                compiled = compile_frame(
                    response,
                    frame,
                    by_id,
                    role_word_limit=config["atomic_role_word_limit"],
                    predicate_word_limit=config["atomic_predicate_word_limit"],
                )
                frames.append(frame)
                questions.extend(compiled["questions"])
                failures.extend(
                    {"proposal": frame, **failure}
                    for failure in compiled["rejected_targets"]
                )
            except (KeyError, TypeError, ValueError) as error:
                failures.append({"proposal": frame, "reason": str(error)})
        marked = raw.get("nonassertion_unit_ids", [])
        for uid in marked if isinstance(marked, list) else []:
            if isinstance(uid, str) and uid in permitted:
                nonassertions.append(by_id[uid]["text"])
    unique = {}
    for question in questions:
        key = (question["claim_quote"], question["answer_quote"], question["question"])
        unique.setdefault(key, question)
    # Round-robin target roles avoids spending the entire fixed cap on only
    # objects or the first long event. No truth scores enter this selection.
    ordered = list(unique.values())
    by_role = {
        role: [q for q in ordered if q["atomic_target_role"] == role]
        for role in sorted(ROLES)
    }
    selected = []
    while len(selected) < question_limit and any(by_role.values()):
        for candidates in by_role.values():
            if candidates and len(selected) < question_limit:
                selected.append(candidates.pop(0))
    return {
        "questions": selected,
        "nonassertion_quotes": nonassertions,
        "atomic_frames": frames,
        "atomic_frame_failures": failures,
        "raw_frame_calls": calls,
        "compiled_questions_total": len(ordered),
        "unselected_questions": max(0, len(ordered) - len(selected)),
        "schema": "atomic-role-question-compiler@2",
        "atomic_budget": {
            "batches_executed": len(calls),
            "frames_considered": considered_frames,
            "returned_frames_omitted_by_budget": omitted_frames,
            "unprocessed_unit_ids": [
                u["id"] for u in units if u["id"] not in processed_units
            ],
            "status": "exhausted"
            if omitted_frames or len(processed_units) < len(units)
            else "complete",
        },
    }


def bind_condition_checks(prediction, question):
    """Validate role-table coverage without changing the reader's raw prediction."""
    result = dict(prediction)
    try:
        conditions = json.loads(question)["conditions"]
        checks = result.get("condition_checks")
        if not isinstance(checks, list):
            raise TypeError("condition_table_missing")
        mapped = {}
        for check in checks:
            if not isinstance(check, dict):
                raise TypeError("condition_row_invalid")
            role = check.get("condition_role")
            if not isinstance(role, str) or role not in conditions or role in mapped:
                raise ValueError("condition_roles_missing_duplicate_or_extra")
            if check.get("condition_quote") != conditions[role]:
                raise ValueError("condition_value_does_not_match_role")
            mapped[role] = check
        if set(mapped) != set(conditions):
            raise ValueError("condition_roles_not_exhaustive")
        result["atomic_condition_binding"] = {
            "valid": True,
            "required_roles": sorted(conditions),
            "all_supported": all(c.get("status") == "supported" for c in checks),
        }
        # Absence of the target is interpretable only after the non-target
        # event conditions themselves have been grounded. Do not turn a
        # missing subject/event premise into a confident target-absence label.
        if not result["atomic_condition_binding"]["all_supported"]:
            for key in ("answerability", "status"):
                if result.get(key) == "not_stated":
                    result["raw_" + key] = result[key]
                    result[key] = "uncertain"
        return result
    except (ValueError, KeyError, TypeError) as error:
        result["atomic_condition_binding"] = {"valid": False, "reason": str(error)}
        result["all_conditions_covered"] = False
        result["all_premises_supported"] = False
        for key in ("answerability", "status"):
            if key in result:
                result["raw_" + key] = result[key]
                result[key] = "uncertain"
        return result


class AtomicReader:
    """Explicit instruction dispatch; actual rewritten prompts use real cache keys."""

    def __init__(self, reader, extraction, protocol=None):
        self.reader, self.extraction = reader, extraction
        self.protocol = PROTOCOL if protocol is None else protocol

    def ask(self, instruction, payload, **kwargs):
        if instruction == prompts.EXTRACT:
            return self.extraction
        replacements = {
            prompts.SELF_QA: MINIMAL_SELF,
            prompts.SOURCE_QA: MINIMAL_SOURCE,
            prompts.VERIFY_SOURCE: MINIMAL_VERIFY,
        }
        if instruction in {prompts.SOURCE_QA, prompts.VERIFY_SOURCE}:
            kwargs["max_new_tokens"] = self.protocol["reader_source_json_token_limit"]
        prediction = self.reader.ask(
            replacements.get(instruction, instruction), payload, **kwargs
        )
        if instruction in {prompts.SOURCE_QA, prompts.VERIFY_SOURCE}:
            prediction = bind_condition_checks(prediction, payload["question"])
        return prediction


def build_atomic_anchors(reader, row, question_limit=48, *, protocol=None):
    extracted = extract_atomic_questions(
        reader, row["response"], question_limit, protocol=protocol
    )
    result = build_anchors(
        AtomicReader(reader, extracted, protocol), row, question_limit
    )
    result["atomic_schema"] = extracted["schema"]
    compiled = {
        (q["claim_quote"], q["answer_quote"], q["question"]): q
        for q in extracted["questions"]
    }
    for question in result["questions"]:
        item = compiled.get(
            tuple(question.get(k) for k in ("claim_quote", "answer_quote", "question"))
        )
        if item is not None:
            for key in (
                "atomic_target_role",
                "atomic_role_bindings",
                "decontext_bindings",
                "question_struct",
                "hidden_answer_aliases",
                "unit_id",
            ):
                question[key] = item[key]
    return result
