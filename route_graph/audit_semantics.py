"""Blinded phase C: fixed candidate text labels, including NULL coverage checks."""

from route_graph import audit_prompts as prompts
from route_graph.evidence_anchor import quote_span
from route_graph.frozen_reader import digest


def label_fixed_pool(
    reader, row, question, frozen, batch_size=12, previous_questions=()
):
    # Position order and opaque text IDs; no native delta, attention, rank,
    # group kind or hypothesis-specific mechanism label enters the reader.
    views = sorted(frozen["views"], key=lambda v: (v["role"], v["keys"][0], v["id"]))
    labels, raw_batches = {}, []
    for begin in range(0, len(views), batch_size):
        batch = views[begin : begin + batch_size]
        output = reader.ask(
            prompts.LABEL_CANDIDATES,
            {
                "source": row["prompt"][slice(*row["source_span"])],
                "question": question["question"],
                "candidates": [
                    {
                        "id": view["id"],
                        "role": view["role"],
                        "segments": view["segments"],
                    }
                    for view in batch
                ],
            },
            max_new_tokens=2048,
        )
        raw_batches.append(output)
        predictions = output.get("candidates", [])
        if not isinstance(predictions, list):
            predictions = []
        for view in batch:
            matching = [
                p
                for p in predictions
                if isinstance(p, dict) and p.get("id") == view["id"]
            ]
            effective = {
                "relation": "uncertain",
                "target_answerability": "uncertain",
                "reason": "missing_or_duplicate_candidate",
            }
            if len(matching) == 1:
                candidate = matching[0]
                relation = candidate.get("relation")
                quotes = candidate.get("evidence_quotes")
                valid = (
                    relation
                    in {
                        "applicable",
                        "contradictory",
                        "nonapplicable",
                        "unrelated",
                        "uncertain",
                    }
                    and isinstance(quotes, list)
                    and all(
                        isinstance(q, str)
                        and q
                        and any(q in s for s in view["segments"])
                        for q in quotes
                    )
                    and (bool(quotes) or relation in {"unrelated", "uncertain"})
                    and candidate.get("target_answerability")
                    in {"answered", "not_stated", "uncertain"}
                )
                if valid:
                    effective = candidate
                else:
                    effective["reason"] = "invalid_candidate_citation_or_schema"
            labels[view["id"]] = effective
    source_units = [
        v for v in views if v["id"] in frozen["text_node_ids"] and v["role"] == "source"
    ]
    null_clear = bool(source_units) and all(
        labels[v["id"]].get("target_answerability") == "not_stated"
        for v in source_units
    )
    relation_check = {"relation": "unknown", "reason": "no_previous_claim"}
    previous = question.get("previous_claim_quote")
    if question.get("previous_claim_span") is not None:
        raw = reader.ask(
            prompts.RELATION_CHECK,
            {
                "previous": previous,
                "current": question["claim_quote"],
                "question": question["question"],
                "current_answer": question["answer_quote"],
                "previous_questions": [
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "answer_quote": q["answer_quote"],
                    }
                    for q in previous_questions
                ],
            },
            max_new_tokens=384,
        )
        relation_check = {**raw, "valid": False}
        try:
            quote_span(previous, raw.get("previous_relation_quote"))
            quote_span(question["claim_quote"], raw.get("current_relation_quote"))
            same = (
                raw.get("relation") in {"same_claim", "elaboration"}
                and raw.get("shared_predicate_or_coreference") is True
            )
            correction = (
                raw.get("relation") == "correction"
                and raw.get("explicit_correction") is True
            )
            relation_check["valid"] = same or correction
            prior = next(
                (
                    q
                    for q in previous_questions
                    if q["id"] == raw.get("previous_question_id")
                ),
                None,
            )
            linked = (
                relation_check["valid"]
                and prior is not None
                and raw.get("current_slot_derives_from_previous_slot") is True
            )
            relation_check["slot_link_valid"] = linked
            if linked:
                relation_check["previous_answer_span"] = prior["answer_span"]
                relation_check["previous_question_id"] = prior["id"]
        except ValueError:
            pass
    return {
        "question_id": question["id"],
        "frozen_pool_sha256": digest(frozen),
        "labels": labels,
        "raw_batches": raw_batches,
        "null_source_unit_check": null_clear,
        "relation_check": relation_check,
        "interface": "fixed_text_only_no_native_effects_or_ranks",
        "prediction_not_gold": True,
    }
