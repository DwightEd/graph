"""Phase A artifact preparation: exact events and all assertion denominators."""

from dataclasses import asdict

from route_graph import audit_prompts as prompts
from route_graph.audit_alignment import align_row, slot_masks, span_keys
from route_graph.audit_protocol import PROTOCOL
from route_graph.causal_groups import build_contrast
from route_graph.cloze_anchor import build_cloze_anchors


def prepare_anchors(reader, tokenizer, row):
    alignment = align_row(row, tokenizer)
    result = build_cloze_anchors(
        reader, row, PROTOCOL["question_limit"], protocol=PROTOCOL
    )
    recoveries = prepare_recoveries(reader, result["questions"])
    for question in result["questions"]:
        if "claim_span" not in question:
            continue
        question["claim_tokens"] = span_keys(
            alignment, "history", question["claim_span"]
        )
        question["answer_tokens"] = span_keys(
            alignment, "history", question["answer_span"]
        )
        for key in ("source_answer", "source_verification"):
            prediction = question.get(key, {})
            prediction["evidence_tokens"] = [
                span_keys(alignment, "source", span)
                for span in prediction.get("evidence_spans", [])
            ]
            checks = prediction.get("condition_checks", [])
            for check in checks if isinstance(checks, list) else []:
                if isinstance(check, dict):
                    check["source_tokens"] = [
                        span_keys(alignment, "source", span)
                        for span in check.get("source_spans", [])
                    ]
        question["events"] = []
        if question.get("contrast", {}).get("status") == "available":
            try:
                for index in range(len(question["contrast"]["edits"])):
                    contrast, metadata = build_contrast(row, question, tokenizer, index)
                    question["events"].append(
                        {
                            "contrast": asdict(contrast),
                            "metadata": metadata,
                            "slot_masks": slot_masks(contrast, metadata, tokenizer),
                        }
                    )
            except (ValueError, StopIteration) as error:
                question["events"] = []
                question["event_error"] = str(error)
    risks = sorted(
        [q for q in result["questions"] if q.get("status") == "unsupported"],
        key=lambda q: (-q["risk"], q["claim_span"][0], q["answer_span"][0], q["id"]),
    )
    # Select by frozen risk and position, even if a selected contrast later fails.
    # Several questions about the same claim consume one claim slot; select its
    # highest-risk question once rather than using all four slots on one sentence.
    selected, seen_spans = [], set()
    for question in risks:
        claim = tuple(question["claim_span"])
        if claim not in seen_spans and len(selected) < PROTOCOL[
            "claims_per_response"
        ] - bool(recoveries):
            selected.append(question["id"])
            seen_spans.add(claim)
    if recoveries:
        selected.append(recoveries[0]["id"])
    for question in result["questions"]:
        if question.get("previous_claim_span") is not None:
            question["previous_answer_candidates"] = [
                {"id": q["id"], "answer_span": q["answer_span"]}
                for q in result["questions"]
                if q.get("claim_span") == question["previous_claim_span"]
                and q.get("id") is not None
                and "answer_span" in q
            ]
    result["native_selected_question_ids"] = selected
    result["alignment"] = alignment
    result["risk_questions_total"] = len(risks)
    result["risk_claims_total"] = len({tuple(q["claim_span"]) for q in risks})
    result["risk_claims_with_valid_contrast"] = len(
        {tuple(q["claim_span"]) for q in risks if q["events"]}
    )
    return result


def prepare_recoveries(reader, questions):
    """An E correction may positively depend on wrong history; audit it as E.

    Reuse a prior contradicted answer only after same-condition verification.
    This is an explicitly wrong alternative, never a fabricated correct answer.
    """
    prepared = []
    for current in questions:
        if (
            current.get("status") != "supported"
            or current.get("previous_claim_span") is None
        ):
            continue
        prior = next(
            (
                q
                for q in questions
                if q.get("claim_span") == current["previous_claim_span"]
                and q.get("effective_scores", {}).get("C", 0) >= 0.8
            ),
            None,
        )
        if prior is None:
            continue
        check = reader.ask(
            prompts.RECOVERY_CHECK,
            {
                "previous_question": prior["question"],
                "current_question": current["question"],
            },
            max_new_tokens=128,
        )
        if check.get("all_conditions_equivalent") is not True:
            current["recovery_check"] = check
            continue
        begin = current["claim_span"][0]
        a, b = current["answer_span"]
        claim = current["claim_quote"]
        replacement = prior["answer_quote"]
        edited = claim[: a - begin] + replacement + claim[b - begin :]
        grammar = reader.ask(
            prompts.CHECK_EDIT,
            {
                "original": claim,
                "edited": edited,
                "target_quote": current["answer_quote"],
                "replacement": replacement,
                "kind": "prior_error_replay: intentionally wrong alternative from the same previous question",
            },
            max_new_tokens=256,
        )
        valid = all(
            grammar.get(key) is True
            for key in ("grammatical", "only_target_changed", "conditions_preserved")
        )
        current["recovery_check"] = {
            **check,
            "prior_question_id": prior["id"],
            "edit_check": grammar,
        }
        if valid:
            current["contrast"] = {
                "status": "available",
                "edits": [
                    {
                        "valid": True,
                        "replacement": replacement,
                        "kind": "supported_vs_prior_error",
                        "check": grammar,
                    }
                ],
            }
            current["audit_role"] = "supported_recovery_control"
            prepared.append(current)
    return sorted(prepared, key=lambda q: (q["claim_span"][0], q["id"]))
