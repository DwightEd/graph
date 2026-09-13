"""Verbatim target masks with exact, integrity-checked condition bindings."""

import hashlib
import json
import re
from itertools import pairwise

MARKER = "<MISSING_SLOT>"


def sidecar_digest(sidecar):
    payload = {k: v for k, v in sidecar.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_sidecar(question, sidecar):
    if sidecar.get("sha256") != sidecar_digest(sidecar):
        raise ValueError("cloze sidecar integrity mismatch")
    if sidecar["question_sha256"] != hashlib.sha256(question.encode()).hexdigest():
        raise ValueError("cloze sidecar belongs to a different question")
    expected = {
        item["role"]: item["span"]
        for item in sidecar["role_bindings"] + sidecar["unassigned_segments"]
        if item["role"] != sidecar["target_role"]
    }
    if set(expected) != set(sidecar["condition_bindings"]):
        raise ValueError("cloze sidecar condition roles differ from raw bindings")
    for role, binding in sidecar["condition_bindings"].items():
        if binding["response_span"] != expected[role]:
            raise ValueError("condition response span differs from its raw role")
        if (
            binding["value"] != sidecar["condition_roles"][role]
            or question[slice(*binding["question_span"])] != binding["value"]
        ):
            raise ValueError("condition sidecar position/value differs from question")


def cloze_candidate(response, compiled):
    """Keep all original unmasked clause text, with explicit non-target referents."""
    if MARKER in response:
        raise ValueError("cloze marker collides with response text")
    claim = compiled["claim_quote"]
    start = response.find(claim)
    if start < 0 or response.find(claim, start + 1) >= 0:
        raise ValueError("ambiguous claim occurrence")
    bindings = {r["role"]: r for r in compiled["atomic_role_bindings"]}
    target = compiled["atomic_target_role"]
    target_span = bindings[target]["span"]
    edits = [(target_span[0] - start, target_span[1] - start, MARKER)]
    conditions = dict(compiled["question_struct"]["conditions"])
    if target in conditions:
        raise ValueError("target is present in condition table")
    for decontext in compiled["decontext_bindings"]:
        role = decontext["role"]
        if role == target:
            continue
        a, b = bindings[role]["span"]
        edits.append((a - start, b - start, decontext["referent_quote"]))
    for a, b, _ in edits:
        if not 0 <= a < b <= len(claim):
            raise ValueError("cloze edit escapes claim")
    ordered = sorted(edits)
    if any(b > next_a for (_, b, _), (next_a, _, _) in pairwise(ordered)):
        raise ValueError("cloze edits overlap")
    text = claim
    for a, b, replacement in reversed(ordered):
        text = text[:a] + replacement + text[b:]
    question = f"Fill {MARKER} in this assertion using the source. Requested role: {target}.\n{text}"
    exposed = question + "\n" + "\n".join(conditions.values())
    for alias in compiled["hidden_answer_aliases"]:
        if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", exposed, re.IGNORECASE):
            raise ValueError("hidden target alias exposed in cloze or conditions")
    # Surface fragments omitted by the role extractor remain explicit required
    # conditions. No stop-word heuristic decides that an unparsed qualifier
    # is irrelevant. Position bindings distinguish repeated short fragments.
    unassigned = []
    cursor = start
    for binding in sorted(bindings.values(), key=lambda r: r["span"]):
        a, b = binding["span"]
        if a > cursor:
            fragment = response[cursor:a].strip()
            if any(char.isalnum() for char in fragment):
                key = f"unparsed_context_{len(unassigned)}"
                conditions[key] = fragment
                raw_fragment = response[cursor:a]
                left = cursor + len(raw_fragment) - len(raw_fragment.lstrip())
                unassigned.append(
                    {
                        "role": key,
                        "quote": fragment,
                        "span": [left, left + len(fragment)],
                    }
                )
        cursor = b
    tail = response[cursor : start + len(claim)].strip()
    if any(char.isalnum() for char in tail):
        key = f"unparsed_context_{len(unassigned)}"
        conditions[key] = tail
        raw_tail = response[cursor : start + len(claim)]
        left = cursor + len(raw_tail) - len(raw_tail.lstrip())
        unassigned.append(
            {"role": key, "quote": tail, "span": [left, left + len(tail)]}
        )
    prefix_length = len(question) - len(text)

    def locate_span(span):
        a, b = span
        left = a - start
        right = b - start
        new_left = left + sum(
            len(value) - (y - x) for x, y, value in edits if y <= left
        )
        new_right = right + sum(
            len(value) - (y - x) for x, y, value in edits if y <= right
        )
        return [prefix_length + new_left, prefix_length + new_right]

    condition_bindings = {}
    for item in [r for role, r in bindings.items() if role != target] + unassigned:
        role = item["role"]
        span = locate_span(item["span"])
        if question[slice(*span)] != conditions[role]:
            raise ValueError("condition does not bind to its exact cloze position")
        condition_bindings[role] = {
            "value": conditions[role],
            "question_span": span,
            "response_span": item["span"],
        }
    sidecar = {
        "masked_claim": text,
        "mask_id": MARKER,
        "target_role": target,
        "role_bindings": compiled["atomic_role_bindings"],
        "condition_roles": conditions,
        "condition_bindings": condition_bindings,
        "decontext_bindings": compiled["decontext_bindings"],
        "hidden_aliases": compiled["hidden_answer_aliases"],
        "unassigned_segments": unassigned,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
    }
    sidecar = json.loads(json.dumps(sidecar))
    sidecar["sha256"] = sidecar_digest(sidecar)
    return {
        **compiled,
        "cloze_sidecar": sidecar,
        "question": question,
        "question_struct": {"requested_role": target, "conditions": conditions},
        "cloze": text,
        "cloze_schema": "verbatim-clause-target-mask@1",
    }


def blind_cloze_source_payload(source, candidate, task):
    """Only source, masked text and NON-target roles cross this interface."""
    target = candidate["atomic_target_role"]
    conditions = candidate["question_struct"]["conditions"]
    if target in conditions or MARKER not in candidate["cloze"]:
        raise ValueError("invalid hidden-target cloze contract")
    visible = {
        "question": candidate["question"],
        "conditions": dict(conditions),
        "task": task,
    }
    serialized = json.dumps(visible, ensure_ascii=False)
    for alias in candidate["hidden_answer_aliases"]:
        if re.search(
            r"(?<!\w)" + re.escape(alias) + r"(?!\w)", serialized, re.IGNORECASE
        ):
            raise ValueError("hidden target alias exposed in source payload")
    sidecar = candidate["cloze_sidecar"]
    validate_sidecar(candidate["question"], sidecar)
    if (
        sidecar["condition_roles"] != conditions
        or sidecar["target_role"] != target
        or sidecar["masked_claim"] != candidate["cloze"]
        or sidecar["role_bindings"] != candidate["atomic_role_bindings"]
        or sidecar["hidden_aliases"] != candidate["hidden_answer_aliases"]
        or sidecar["decontext_bindings"] != candidate["decontext_bindings"]
    ):
        raise ValueError("cloze sidecar differs from compiled candidate")
    return {"source": source, **visible}


def bind_cloze_condition(question, sidecar, check):
    """Resolve repeated condition text by role/position, never substring guessing."""
    validate_sidecar(question, sidecar)
    role = check.get("condition_role")
    binding = sidecar["condition_bindings"].get(role)
    if binding is None or check.get("condition_quote") != binding["value"]:
        raise ValueError("condition role/value differs from the cloze sidecar")
    span = binding["question_span"]
    if question[slice(*span)] != binding["value"]:
        raise ValueError("condition sidecar position differs from the actual question")
    return list(span)


class ClozeReader:
    """Separate current-assertion recoverability from blind source verification."""

    def __init__(self, reader, extraction, protocol):
        self.reader, self.extraction, self.protocol = reader, extraction, protocol
        self.candidates = {q["question"]: q for q in extraction["questions"]}

    def ask(self, instruction, payload, **kwargs):
        import json

        from route_graph import audit_prompts as prompts
        from route_graph import cloze_prompts
        from route_graph.atomic_anchor import bind_condition_checks

        if instruction == prompts.EXTRACT:
            return self.extraction
        if instruction == prompts.SELF_QA:
            return self.reader.ask(cloze_prompts.SELF, payload, **kwargs)
        if instruction in {prompts.SOURCE_QA, prompts.VERIFY_SOURCE}:
            if set(payload) != {"source", "question", "task"}:
                raise ValueError("unexpected fields in blind source request")
            candidate = self.candidates[payload["question"]]
            blind = blind_cloze_source_payload(
                payload["source"], candidate, payload["task"]
            )
            actual = (
                cloze_prompts.SOURCE
                if instruction == prompts.SOURCE_QA
                else cloze_prompts.VERIFY
            )
            kwargs["max_new_tokens"] = self.protocol["reader_source_json_token_limit"]
            prediction = self.reader.ask(actual, blind, **kwargs)
            return bind_condition_checks(
                prediction, json.dumps(candidate["question_struct"])
            )
        return self.reader.ask(instruction, payload, **kwargs)


def build_cloze_anchors(reader, row, question_limit=48, *, protocol=None):
    from route_graph.atomic_anchor import extract_atomic_questions
    from route_graph.audit_protocol import PROTOCOL
    from route_graph.evidence_anchor import build_anchors

    config = PROTOCOL if protocol is None else protocol
    extracted = extract_atomic_questions(
        reader, row["response"], question_limit, protocol=config
    )
    candidates, failures = [], []
    for compiled in extracted["questions"]:
        try:
            candidates.append(cloze_candidate(row["response"], compiled))
        except (ValueError, TypeError, KeyError) as error:
            failures.append({"proposal": compiled, "reason": str(error)})
    extracted = {
        **extracted,
        "questions": candidates,
        "cloze_failures": failures,
        "schema": "verbatim-clause-target-mask@1",
    }
    result = build_anchors(ClozeReader(reader, extracted, config), row, question_limit)
    result["atomic_schema"] = extracted["schema"]
    by_question = {q["question"]: q for q in candidates}
    for question in result["questions"]:
        compiled = by_question.get(question.get("question"))
        if compiled is not None:
            for key in (
                "atomic_target_role",
                "atomic_role_bindings",
                "decontext_bindings",
                "question_struct",
                "hidden_answer_aliases",
                "unit_id",
                "cloze",
                "cloze_schema",
                "cloze_sidecar",
            ):
                question[key] = compiled[key]
    return result
