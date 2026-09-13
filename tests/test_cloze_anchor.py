import json

import pytest

from route_graph.atomic_anchor import compile_frame, response_units
from route_graph.cloze_anchor import MARKER, blind_cloze_source_payload, cloze_candidate


def candidate(target="destination"):
    response = (
        "Before dawn, the crane carefully moved the crate from the yard to the dock."
    )
    frame = {
        "unit_id": "u0",
        "anchor_quote": response,
        "roles": [
            {"role": "subject", "quote": "the crane"},
            {"role": "predicate", "quote": "moved"},
            {"role": "object", "quote": "the crate"},
            {"role": "origin", "quote": "the yard"},
            {"role": "destination", "quote": "the dock"},
        ],
    }
    compiled = compile_frame(
        response, frame, {u["id"]: u for u in response_units(response)}
    )
    q = next(q for q in compiled["questions"] if q["atomic_target_role"] == target)
    return response, q


def test_unassigned_qualifiers_survive_verbatim_target_mask():
    response, q = candidate()
    cloze = cloze_candidate(response, q)
    assert cloze["cloze"] == response.replace("the dock", MARKER)
    assert "Before dawn" in cloze["question"] and "carefully" in cloze["question"]
    assert "the dock" not in cloze["question"]


def test_source_payload_does_not_contain_unmasked_claim_or_target_fields():
    response, q = candidate()
    cloze = cloze_candidate(response, q)
    payload = blind_cloze_source_payload("A separate source.", cloze, "QA")
    assert set(payload) == {"source", "question", "conditions", "task"}
    serialized = json.dumps(payload)
    assert response not in serialized and "the dock" not in serialized
    assert "destination" not in payload["conditions"]


def test_source_may_naturally_contain_the_answer_without_being_stripped():
    response, q = candidate()
    cloze = cloze_candidate(response, q)
    payload = blind_cloze_source_payload(response, cloze, "QA")
    assert payload["source"] == response


def test_original_unmasked_context_cannot_reintroduce_a_target_alias():
    response, q = candidate("subject")
    q["hidden_answer_aliases"].append("Before dawn")
    with pytest.raises(ValueError, match="alias"):
        cloze_candidate(response, q)


def test_unparsed_qualifiers_are_explicit_sidecar_conditions():
    response, q = candidate()
    cloze = cloze_candidate(response, q)
    sidecar = cloze["cloze_sidecar"]
    assert sidecar["mask_id"] == MARKER
    assert sidecar["masked_claim"] == cloze["cloze"]
    assert "Before dawn," in sidecar["condition_roles"].values()
    assert "carefully" in sidecar["condition_roles"].values()
    assert sidecar["role_bindings"] == q["atomic_role_bindings"]
    assert all(
        response[slice(*s["span"])] == s["quote"]
        for s in sidecar["unassigned_segments"]
    )


def test_blind_payload_rechecks_all_visible_fields():
    response, q = candidate()
    cloze = cloze_candidate(response, q)
    cloze["question_struct"]["conditions"]["unparsed_context_extra"] = "the dock"
    with pytest.raises(ValueError, match="alias"):
        blind_cloze_source_payload("source", cloze, "QA")


def test_condition_roles_bind_exact_question_positions_including_unparsed_text():
    from route_graph.cloze_anchor import bind_cloze_condition

    response, q = candidate()
    cloze = cloze_candidate(response, q)
    sidecar = cloze["cloze_sidecar"]
    for role, value in sidecar["condition_roles"].items():
        span = bind_cloze_condition(
            cloze["question"],
            sidecar,
            {"condition_role": role, "condition_quote": value},
        )
        assert cloze["question"][slice(*span)] == value
    with pytest.raises(ValueError, match="role/value"):
        bind_cloze_condition(
            cloze["question"],
            sidecar,
            {"condition_role": "origin", "condition_quote": "the dock"},
        )


def test_repeated_referent_values_keep_distinct_role_coordinates():
    from route_graph.cloze_anchor import bind_cloze_condition

    response = "The crane waited. It observed itself."
    frame = {
        "unit_id": "u1",
        "anchor_quote": "It observed itself.",
        "roles": [
            {"role": "subject", "quote": "It", "referent_quote": "The crane"},
            {"role": "predicate", "quote": "observed"},
            {"role": "object", "quote": "itself", "referent_quote": "The crane"},
        ],
    }
    compiled = compile_frame(
        response, frame, {u["id"]: u for u in response_units(response)}
    )
    q = next(q for q in compiled["questions"] if q["atomic_target_role"] == "predicate")
    cloze = cloze_candidate(response, q)
    spans = [
        bind_cloze_condition(
            cloze["question"],
            cloze["cloze_sidecar"],
            {"condition_role": role, "condition_quote": "The crane"},
        )
        for role in ("subject", "object")
    ]
    assert spans[0] != spans[1]
    assert all(cloze["question"][slice(*s)] == "The crane" for s in spans)
    cloze["cloze_sidecar"]["condition_bindings"]["subject"]["question_span"] = spans[1]
    with pytest.raises(ValueError, match="integrity"):
        bind_cloze_condition(
            cloze["question"], cloze["cloze_sidecar"],
            {"condition_role": "subject", "condition_quote": "The crane"},
        )
    with pytest.raises(ValueError, match="integrity"):
        blind_cloze_source_payload("source", cloze, "QA")


def test_source_anchor_uses_role_coordinates_and_never_receives_unmasked_claim():
    from route_graph import cloze_prompts
    from route_graph.audit_protocol import PROTOCOL
    from route_graph.cloze_anchor import ClozeReader
    from route_graph.evidence_anchor import _source_anchor

    response = "The crane waited. It observed itself."
    source = "The crane observed itself."
    frame = {
        "unit_id": "u1",
        "anchor_quote": "It observed itself.",
        "roles": [
            {"role": "subject", "quote": "It", "referent_quote": "The crane"},
            {"role": "predicate", "quote": "observed"},
            {"role": "object", "quote": "itself", "referent_quote": "The crane"},
        ],
    }
    compiled = compile_frame(
        response, frame, {u["id"]: u for u in response_units(response)}
    )
    q = cloze_candidate(
        response,
        next(
            q for q in compiled["questions"] if q["atomic_target_role"] == "predicate"
        ),
    )

    class Reader:
        def ask(self, instruction, payload, **kwargs):
            assert instruction in {cloze_prompts.SOURCE, cloze_prompts.VERIFY}
            assert set(payload) == {"source", "question", "conditions", "task"}
            assert payload["source"] == source
            assert "observed" not in json.dumps(
                {k: v for k, v in payload.items() if k != "source"}
            )
            return {
                "answerability": "answerable",
                "status": "supported",
                "source_answer": "observed",
                "source_answer_quote": "observed",
                "evidence_quotes": [source],
                "all_conditions_covered": True,
                "all_premises_supported": True,
                "unresolved_candidates": [],
                "condition_checks": [
                    {
                        "condition_role": role,
                        "condition_quote": value,
                        "source_quotes": [source],
                        "status": "supported",
                    }
                    for role, value in payload["conditions"].items()
                ],
            }

    answer, verification = _source_anchor(
        ClozeReader(Reader(), {"questions": [q]}, PROTOCOL),
        source,
        q["question"],
        "QA",
        cloze_sidecar=q["cloze_sidecar"],
    )
    assert verification["status"] == "supported"
    assert "citation_error" not in answer and "citation_error" not in verification
    assert (
        answer["condition_checks"][0]["question_span"]
        != answer["condition_checks"][1]["question_span"]
    )
