"""Automatic relation questions and fallible source anchors, without gold labels.

The source QA request accepts only source and question. Original response values
are introduced again only in the comparison step. Every failure retains a record.
"""

import re

from route_graph import audit_prompts as prompts
from route_graph.frozen_reader import digest


def quote_span(text, quote, within=None):
    if not isinstance(quote, str) or not quote:
        raise ValueError("empty or invalid quote")
    left, right = within or (0, len(text))
    first = text.find(quote, left, right)
    if first < 0 or text.find(quote, first + 1, right) >= 0:
        raise ValueError("quote absent or ambiguous")
    return [first, first + len(quote)]


def source_qa_request(source, question, task="unspecified"):
    """This interface cannot accidentally copy the response answer or risk."""
    if not isinstance(source, str) or not isinstance(question, str):
        raise TypeError("source and question must be strings")
    return {"source": source, "question": question, "task": task}


def text_units(text):
    """Cover all characters; boundaries group text, never truncate context."""
    ends = [m.end() for m in re.finditer(r"[.!?;](?=\s|$)|\n", text)]
    if not ends or ends[-1] < len(text):
        ends.append(len(text))
    left, units = 0, []
    for right in ends:
        if right > left:
            units.append({"start": left, "end": right, "text": text[left:right]})
        left = right
    return units


def _quotes_valid(source, value):
    if not isinstance(value, list) or any(
        not isinstance(q, str) or not q for q in value
    ):
        return False
    # Repeated source quotes remain legitimate sets of possible positions.
    return all(q in source for q in value)


def _candidate_question(response, item):
    if not isinstance(item, dict):
        raise TypeError("question must be an object")
    claim = quote_span(response, item.get("claim_quote"))
    answer = quote_span(response, item.get("answer_quote"), claim)
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question missing")
    answer_quote = item["answer_quote"].strip()
    if re.search(
        r"(?<!\w)" + re.escape(answer_quote) + r"(?!\w)", question, re.IGNORECASE
    ):
        raise ValueError("answer appears in question")
    premises = item.get("premise_quotes")
    if not isinstance(premises, list):
        raise TypeError("premises must be exact response quotes")
    premise_spans = [
        quote_span(
            response, value, claim if value in item["claim_quote"] else (0, claim[0])
        )
        for value in premises
    ]
    if any(a < answer[1] and b > answer[0] for a, b in premise_spans):
        raise ValueError("question premise overlaps hidden answer")
    relation = item.get("relation_to_previous", "unknown")
    previous = item.get("previous_claim_quote")
    previous_span = None
    if relation not in {
        "same_claim",
        "elaboration",
        "correction",
        "new_claim",
        "unknown",
    }:
        relation = "unknown"
    if relation in {"same_claim", "elaboration", "correction"}:
        try:
            previous_span = quote_span(response, previous, (0, claim[0]))
        except ValueError:
            relation = "unknown"
    result = {
        "id": digest({"claim": claim, "answer": answer, "question": question})[:20],
        "claim_span": claim,
        "answer_span": answer,
        "claim_quote": item["claim_quote"],
        "answer_quote": item["answer_quote"],
        "question": question,
        "premise_spans": premise_spans,
        "slot_type": item.get("slot_type", "other"),
        "predicate": item.get("predicate", "unknown"),
        "relation_to_previous": relation,
        "previous_claim_quote": previous,
        "previous_claim_span": previous_span,
    }
    if "cloze_sidecar" in item:
        result["cloze_sidecar"] = item["cloze_sidecar"]
    return result


def _check_question(reader, response, question):
    payload = {"response": response, "question": question["question"]}
    if "cloze_sidecar" in question:
        payload["current_assertion"] = question["claim_quote"]
    result = reader.ask(
        prompts.SELF_QA,
        payload,
        max_new_tokens=384,
    )
    valid = (
        result.get("question_faithful") is True
        and result.get("all_conditions_preserved") is True
        and result.get("answer_leaked_in_question") is False
        and result.get("answer_quote") == question["answer_quote"]
    )
    return result, valid


def _source_anchor(reader, source, question, task="unspecified", *, cloze_sidecar=None):
    answer = reader.ask(
        prompts.SOURCE_QA, source_qa_request(source, question, task), max_new_tokens=768
    )
    valid_quotes = _quotes_valid(source, answer.get("evidence_quotes"))
    if not valid_quotes or "reader_error" in answer:
        return answer, {"status": "uncertain", "reason": "invalid_source_quotes"}
    verification = reader.ask(
        prompts.VERIFY_SOURCE,
        source_qa_request(source, question, task),
        max_new_tokens=768,
    )
    if not _quotes_valid(source, verification.get("evidence_quotes")):
        verification = {
            **verification,
            "status": "uncertain",
            "reason": "invalid_verification_quotes",
        }
    # Ambiguous string citations cannot identify event ownership. Keep the raw
    # prediction, but move the effective verification to uncertain.
    for prediction in (answer, verification):
        try:
            prediction["evidence_spans"] = [
                quote_span(source, q) for q in prediction.get("evidence_quotes", [])
            ]
            quoted_answer = prediction.get("source_answer_quote")
            if quoted_answer is not None:
                located = set()
                if not isinstance(quoted_answer, str) or not quoted_answer:
                    raise ValueError("invalid source answer quote")
                for left, right in prediction["evidence_spans"]:
                    begin = source.find(quoted_answer, left, right)
                    while begin >= 0:
                        located.add((begin, begin + len(quoted_answer)))
                        begin = source.find(quoted_answer, begin + 1, right)
                if len(located) != 1:
                    raise ValueError(
                        "source answer absent or ambiguous within cited evidence"
                    )
                prediction["source_answer_span"] = list(next(iter(located)))
            checks = prediction.get("condition_checks", [])
            if not isinstance(checks, list):
                raise TypeError("condition table must be a list")
            for check in checks:
                if not isinstance(check, dict) or check.get("status") not in {
                    "supported",
                    "missing",
                    "conflicting",
                    "uncertain",
                }:
                    raise ValueError("invalid condition check")
                if cloze_sidecar is None:
                    check["question_span"] = quote_span(
                        question, check.get("condition_quote")
                    )
                else:
                    from route_graph.cloze_anchor import bind_cloze_condition

                    check["question_span"] = bind_cloze_condition(
                        question, cloze_sidecar, check
                    )
                check["source_spans"] = [
                    quote_span(source, q) for q in check.get("source_quotes", [])
                ]
                if check["status"] == "supported" and not check["source_spans"]:
                    raise ValueError("supported condition has no citation")
            prediction["condition_table_valid"] = bool(checks)
            if "all_premises_supported" in prediction:
                prediction["all_premises_supported"] = (
                    prediction["all_premises_supported"] is True
                    and bool(checks)
                    and all(check["status"] == "supported" for check in checks)
                )
        except (ValueError, TypeError):
            prediction["citation_error"] = "ambiguous_source_occurrence"
            verification = {**verification, "status": "uncertain"}
    return answer, verification


def _classify(reader, question, answer, verification):
    raw = reader.ask(
        prompts.COMPARE,
        {
            "question": question["question"],
            "response_answer": question["answer_quote"],
            "source_answer": answer,
            "source_verification": verification,
        },
        labels=["E", "C", "N", "U"],
        max_new_tokens=1,
    )
    if "reader_error" in raw:
        return raw, {"E": 0.0, "C": 0.0, "N": 0.0, "U": 1.0}
    scores = dict(zip(raw["labels"], raw["probabilities"], strict=True))
    complete = (
        answer.get("answerability") == "answerable"
        and answer.get("all_conditions_covered") is True
        and verification.get("status") == "supported"
        and verification.get("all_conditions_covered") is True
        and verification.get("all_premises_supported") is True
        and bool(answer.get("evidence_quotes"))
        and bool(verification.get("evidence_quotes"))
        and answer.get("source_answer_quote") is not None
        and answer.get("source_answer_quote") == verification.get("source_answer_quote")
        and answer.get("condition_table_valid") is True
        and verification.get("condition_table_valid") is True
        and all(
            check["status"] == "supported"
            for prediction in (answer, verification)
            for check in prediction["condition_checks"]
        )
        and not answer.get("unresolved_candidates", ["missing"])
        and not verification.get("unresolved_candidates", ["missing"])
    )
    missing = (
        answer.get("answerability") == "not_stated"
        and verification.get("status") == "not_stated"
        and not answer.get("unresolved_candidates", ["missing"])
        and not verification.get("unresolved_candidates", ["missing"])
    )
    # Move invalid semantic classes to U without renormalizing/inflating risk.
    for label, permitted in (("E", complete), ("C", complete), ("N", missing)):
        if not permitted:
            scores["U"] += scores[label]
            scores[label] = 0.0
    return raw, scores


def _make_edits(reader, source, question, answer, scores):
    if scores["C"] >= 0.8:
        replacement = answer.get("source_answer_quote")
        if (
            not isinstance(replacement, str)
            or not replacement
            or replacement not in source
            or replacement != answer.get("source_answer")
            or not any(
                replacement in quote for quote in answer.get("evidence_quotes", [])
            )
        ):
            return {"status": "contrast_requires_rewrite", "edits": []}
        kind, replacements = "grounded_replacement", [replacement]
    elif scores["N"] >= 0.8:
        kind = "commitment_vs_withholding"
        replacements = prompts.WITHHOLDING.get(question["slot_type"], ())
        if not replacements:
            return {"status": "withholding_type_unavailable", "edits": []}
    else:
        return {"status": "no_high_risk_contrast", "edits": []}
    edits = []
    a, b = question["answer_span"]
    begin = question["claim_span"][0]
    claim = question["claim_quote"]
    for replacement in replacements:
        changed = claim[: a - begin] + replacement + claim[b - begin :]
        checked = reader.ask(
            prompts.CHECK_EDIT,
            {
                "original": claim,
                "edited": changed,
                "target_quote": question["answer_quote"],
                "replacement": replacement,
                "kind": kind,
            },
            max_new_tokens=256,
        )
        valid = all(
            checked.get(k) is True
            for k in ("grammatical", "only_target_changed", "conditions_preserved")
        )
        edits.append(
            {"replacement": replacement, "kind": kind, "check": checked, "valid": valid}
        )
    return {
        "status": "available" if all(e["valid"] for e in edits) else "invalid_edit",
        "edits": edits,
    }


def build_anchors(reader, row, question_limit=48):
    """Return full-coverage semantic artifacts; no natural labels are accepted."""
    response = row["response"]
    source = row["prompt"][slice(*row["source_span"])]
    extracted = reader.ask(
        prompts.EXTRACT,
        {
            "task": row["task"],
            "response": response,
            "question_limit": question_limit,
        },
        max_new_tokens=3072,
    )
    proposed = extracted.get("questions", [])
    if not isinstance(proposed, list):
        proposed = []
    questions, seen = [], set()
    for item in proposed[:question_limit]:
        try:
            question = _candidate_question(response, item)
        except (ValueError, KeyError, TypeError) as error:
            questions.append(
                {"status": "invalid_question", "reason": str(error), "proposal": item}
            )
            continue
        if question["id"] in seen:
            continue
        seen.add(question["id"])
        checked, valid = _check_question(reader, response, question)
        question["self_check"] = checked
        if not valid:
            questions.append({**question, "status": "invalid_question", "risk": 0.5})
            continue
        answer, verification = _source_anchor(
            reader,
            source,
            question["question"],
            row["task"],
            cloze_sidecar=question.get("cloze_sidecar"),
        )
        raw, scores = _classify(reader, question, answer, verification)
        risk = scores["C"] + scores["N"]
        supported = scores["E"] >= 0.8
        unsupported = risk >= 0.8
        status = (
            "unsupported"
            if unsupported
            else ("supported" if supported else "uncertain")
        )
        localization = (
            "slot" if verification.get("all_premises_supported") is True else "claim"
        )
        question.update(
            status=status,
            source_answer=answer,
            source_verification=verification,
            label_scores=raw,
            effective_scores=scores,
            risk=risk,
            localization=localization,
            null_scope="reader_estimate" if scores["N"] else None,
            contrast=_make_edits(reader, source, question, answer, scores),
        )
        questions.append(question)
    nonassertions, invalid_nonassertions = [], []
    raw_nonassertions = extracted.get("nonassertion_quotes", [])
    if not isinstance(raw_nonassertions, list):
        raw_nonassertions = []
    for quote in raw_nonassertions:
        try:
            nonassertions.append(quote_span(response, quote))
        except ValueError:
            invalid_nonassertions.append(quote)
    words = []
    for word in re.finditer(r"\S+", response):
        covering = [
            q
            for q in questions
            if "claim_span" in q
            and q["claim_span"][0] < word.end()
            and q["claim_span"][1] > word.start()
        ]
        valid = []
        for question in covering:
            if question["status"] not in {"supported", "unsupported"}:
                continue
            scope = question["claim_span"]
            if (
                question["status"] == "unsupported"
                and question["localization"] == "slot"
            ):
                scope = question["answer_span"]
            if scope[0] < word.end() and scope[1] > word.start():
                valid.append(question)
        words.append(
            {
                "start": word.start(),
                "end": word.end(),
                "score": max((q["risk"] for q in valid), default=0.5),
                "abstain": not bool(valid),
                "question_ids": [q["id"] for q in covering],
                "coverage_state": "assertion"
                if covering
                else (
                    "nonassertion"
                    if any(
                        a <= word.start() and b >= word.end() for a, b in nonassertions
                    )
                    else "coverage_unknown"
                ),
            }
        )
    return {
        "schema": "native-audit/anchors@1",
        "id": row["id"],
        "source_id": row["source_id"],
        "response_sha256": row["response_sha256"],
        "extraction": extracted,
        "questions": questions,
        "words": words,
        "nonassertion_spans": nonassertions,
        "invalid_nonassertions": invalid_nonassertions,
        "coverage": {
            "words": len(words),
            "scored_words": sum(not w["abstain"] for w in words),
            "proposed_questions": len(proposed),
            "retained_questions": len(questions),
        },
        "semantic_status": "frozen_reader_estimate_not_gold",
    }
