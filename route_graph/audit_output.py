"""Merge factual estimates, native witnesses and conditional history edges."""

import re
from collections import Counter


def merge_response(row, anchors, proposals, semantics, validations):
    questions = anchors["questions"]
    states, edges, claim_results = {}, [], []
    for question in questions:
        qid = question.get("id")
        if qid is None:
            continue
        sem, native = semantics.get(qid, {}), validations.get(qid, {})
        missing = question.get("effective_scores", {}).get("N", 0) >= 0.8
        semantic_valid = question.get("status") in {"supported", "unsupported"}
        if missing and not sem.get("null_source_unit_check", False):
            semantic_valid = False
        positions = [p for p in native.get("positions", []) if p.get("passed")]
        raw_position_ids = [p["id"] for p in positions]
        origins = native.get("origins", {})
        template = native.get("template", {})
        if missing:
            positions = [
                p
                for p in positions
                if template.get("passed")
                and template.get("origin_passed")
                and origins.get(p["id"], {}).get("passed")
                and p["id"] == template.get("primary_id")
            ]
        mechanisms = []
        for position in positions:
            origin = origins.get(position["id"], {})
            mediated = origin.get("passed") and (
                not missing or template.get("origin_passed")
            )
            group = position["group"]
            if group["kind"] != "content" or not mediated:
                continue
            if (
                group["role"] == "source"
                and position["delta"] > 0
                and position["semantic"].get("relation") == "nonapplicable"
            ):
                mechanisms.append("nonapplicable_source_adoption")
            if group["role"] == "history":
                relation = sem.get("relation_check", {})
                previous = [
                    p
                    for p in questions
                    if p.get("claim_span") == question.get("previous_claim_span")
                    and p.get("status") == "unsupported"
                    and p.get("id") == relation.get("previous_question_id")
                ]
                previous_wrong = any(
                    p.get("effective_scores", {}).get("C", 0) >= 0.8
                    or (
                        p.get("effective_scores", {}).get("N", 0) >= 0.8
                        and semantics.get(p["id"], {}).get(
                            "null_source_unit_check", False
                        )
                    )
                    for p in previous
                )
                current_wrong = (
                    question.get("status") == "unsupported" and semantic_valid
                )
                valid_relation = (
                    relation.get("valid") is True
                    and relation.get("slot_link_valid") is True
                    and origin.get("origin_selection") == "mapped_previous_answer_slot"
                )
                continues = (
                    previous_wrong
                    and current_wrong
                    and valid_relation
                    and relation.get("relation") in {"same_claim", "elaboration"}
                    and position["delta"] > 0
                )
                edge_type = "conditional_history_dependency"
                if continues:
                    edge_type = "estimated_error_continuation_with_mediated_dependency"
                    mechanisms.append("error_history_continuation")
                elif (
                    question.get("status") == "supported"
                    and previous_wrong
                    and valid_relation
                ):
                    edge_type = "supported_recovery_dependency"
                edges.append(
                    {
                        "from_span": question.get("previous_claim_span"),
                        "from_answer_span": relation.get("previous_answer_span"),
                        "from_question_id": relation.get("previous_question_id"),
                        "from_token_positions": origin["origin"],
                        "to_span": question["claim_span"],
                        "type": edge_type,
                        "position_delta": position["delta"],
                        "input_mediated_delta": origin["delta"],
                        "relation_prediction": relation,
                        "target_question_id": qid,
                        "scope": "measured_current_event_only_not_all_intervening_words",
                    }
                )
        if not missing:
            pair = native.get("paired_role", {})
            if pair.get("passed") and pair.get("origin_scope_matched"):
                mechanisms.append("selective_history_to_applicable_source_transport")
            interaction = native.get("MLP_interaction", {})
            if (
                interaction.get("passed")
                and interaction.get("status") == "evidence_MLP_antagonism"
            ):
                mechanisms.append("evidence_MLP_antagonism")
        resolved = semantic_valid and bool(mechanisms)
        states[qid] = {
            "semantic_valid": semantic_valid,
            "mechanism_resolved": resolved,
            "internal_certificate": bool(positions),
            "risk": question.get("risk", 0.5),
        }
        claim_results.append(
            {
                "id": qid,
                "claim_span": question["claim_span"],
                "answer_span": question["answer_span"],
                "semantic_state": question["status"],
                "semantic_valid": semantic_valid,
                "native_status": native.get(
                    "status", "not_selected_or_contrast_unavailable"
                ),
                "raw_position_certificate_ids": raw_position_ids,
                "native_certificate_ids": [p["id"] for p in positions],
                "internal_certificate_scope": "two_template_position_and_origin"
                if missing
                else "finite_position_effect",
                "mechanisms": mechanisms,
                "boundary_prediction": question.get("relation_to_previous", "unknown"),
                "audit_role": question.get("audit_role", "risk_claim"),
            }
        )
    words = []
    coverage_states = {
        (w["start"], w["end"]): w["coverage_state"] for w in anchors["words"]
    }
    for word in re.finditer(r"\S+", row["response"]):
        covered = []
        for question in questions:
            qid = question.get("id")
            if qid not in states or not states[qid]["semantic_valid"]:
                continue
            scope = question["claim_span"]
            if (
                question["status"] == "unsupported"
                and question["localization"] == "slot"
            ):
                scope = question["answer_span"]
            if scope[0] < word.end() and scope[1] > word.start():
                covered.append(states[qid])
        mechanism_scores = [s["risk"] for s in covered if s["mechanism_resolved"]]
        words.append(
            {
                "start": word.start(),
                "end": word.end(),
                "coverage_state": coverage_states[(word.start(), word.end())],
                "semantic_score": max((s["risk"] for s in covered), default=0.5),
                "semantic_abstain": not bool(covered),
                "mechanism_score": max(mechanism_scores, default=0.5),
                "mechanism_abstain": not bool(mechanism_scores),
            }
        )
    # Semantic segmentation proposes boundaries; measured cross-span edges are
    # retained explicitly instead of hard-truncating information at a sentence.
    graph = {
        "claim_nodes": claim_results,
        "conditional_history_edges": edges,
        "native_groups": [
            {"question_id": qid, "groups": p.get("frozen", {}).get("finalists", [])}
            for qid, p in proposals.items()
        ],
        "origin_artifacts": [
            {"question_id": qid, "artifacts": v.get("donor_artifacts", [])}
            for qid, v in validations.items()
        ],
        "unique_lookback_node_claim": False,
    }
    risk_spans = {
        tuple(q["claim_span"]) for q in questions if q.get("status") == "unsupported"
    }
    recovery_spans = {
        tuple(q["claim_span"])
        for q in questions
        if q.get("audit_role") == "supported_recovery_control"
    }

    def counted_spans(field, allowed):
        return len(
            {
                tuple(q["claim_span"])
                for q in questions
                if q.get("id") in states
                and states[q["id"]][field]
                and tuple(q["claim_span"]) in allowed
            }
        )

    reader_outcomes = Counter(anchors.get("reader_outcomes", {}))
    for semantic in semantics.values():
        reader_outcomes.update(semantic.get("reader_outcomes", {}))

    return {
        "schema": "native-audit/result@1",
        "id": row["id"],
        "source_id": row["source_id"],
        "task": row["task"],
        "generator": row["generator"],
        "response_sha256": row["response_sha256"],
        "words": words,
        "semantic_A_words": anchors["words"],
        "graph": graph,
        "reader_request_accounting": dict(reader_outcomes),
        "coverage": {
            "word_states": dict(Counter(w["coverage_state"] for w in words)),
            "all_words": len(words),
            "semantic_scored_words": sum(not w["semantic_abstain"] for w in words),
            "mechanism_scored_words": sum(not w["mechanism_abstain"] for w in words),
            "risk_claims": anchors["risk_claims_total"],
            "risk_claims_with_valid_contrast": anchors[
                "risk_claims_with_valid_contrast"
            ],
            "risk_claims_with_internal_certificate": counted_spans(
                "internal_certificate", risk_spans
            ),
            "risk_claims_with_resolved_mechanism": counted_spans(
                "mechanism_resolved", risk_spans
            ),
            "recovery_control_claims": len(recovery_spans),
            "recovery_claims_with_internal_certificate": counted_spans(
                "internal_certificate", recovery_spans
            ),
            "recovery_claims_with_resolved_mechanism": counted_spans(
                "mechanism_resolved", recovery_spans
            ),
        },
        "forward_calls": sum(v.get("forward_calls", 0) for v in validations.values()),
        "native_tokens_processed": sum(
            v.get("tokens_processed", 0) for v in validations.values()
        ),
        "claim_supported": "not_evaluated_against_independent_labels",
    }
