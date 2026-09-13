import json

import pytest

from route_graph.atomic_anchor import (
    bind_condition_checks,
    compile_frame,
    response_units,
)
from route_graph.evidence_anchor import _candidate_question


def frame():
    return {
        "unit_id": "u0",
        "anchor_quote": "The crane moved the crate from the yard to the dock.",
        "roles": [
            {"role": "subject", "quote": "The crane"},
            {"role": "predicate", "quote": "moved"},
            {"role": "object", "quote": "the crate"},
            {"role": "origin", "quote": "the yard"},
            {"role": "destination", "quote": "the dock"},
        ],
    }


def test_compiler_hides_each_role_and_preserves_all_other_event_conditions():
    item = frame()
    response = item["anchor_quote"]
    questions = compile_frame(
        response, item, {u["id"]: u for u in response_units(response)}
    )["questions"]
    assert len(questions) == 5
    for question in questions:
        spec = json.loads(question["question"])
        target = spec["requested_role"]
        assert target not in spec["conditions"]
        assert len(spec["conditions"]) == 4
        assert question["answer_quote"] not in question["question"]
        checked = _candidate_question(response, question)
        assert response[slice(*checked["answer_span"])] == question["answer_quote"]
        assert all(
            not (a < checked["answer_span"][1] and b > checked["answer_span"][0])
            for a, b in checked["premise_spans"]
        )


def test_compiler_rejects_role_overlap_instead_of_later_hiding_a_premise():
    item = frame()
    item["roles"][1]["quote"] = "moved the crate"
    response = item["anchor_quote"]
    with pytest.raises(ValueError, match="overlap"):
        compile_frame(response, item, {u["id"]: u for u in response_units(response)})


def test_decontextualization_keeps_the_antecedent_out_of_its_own_hidden_role():
    response = "The crane stood outside. It moved the crate."
    units = {u["id"]: u for u in response_units(response)}
    item = {
        "unit_id": "u1",
        "anchor_quote": "It moved the crate.",
        "roles": [
            {"role": "subject", "quote": "It", "referent_quote": "The crane"},
            {"role": "predicate", "quote": "moved"},
            {"role": "object", "quote": "the crate"},
        ],
    }
    questions = compile_frame(response, item, units)["questions"]
    subject = next(q for q in questions if q["atomic_target_role"] == "subject")
    predicate = next(q for q in questions if q["atomic_target_role"] == "predicate")
    assert "The crane" not in subject["question"]
    assert json.loads(predicate["question"])["conditions"]["subject"] == "The crane"
    bound = _candidate_question(response, predicate)
    assert [0, 9] in bound["premise_spans"]


def test_reference_cannot_be_imported_from_future_response_text():
    response = "It moved the crate. The crane stood outside."
    units = {u["id"]: u for u in response_units(response)}
    item = {
        "unit_id": "u0",
        "anchor_quote": "It moved the crate.",
        "roles": [
            {"role": "subject", "quote": "It", "referent_quote": "The crane"},
            {"role": "predicate", "quote": "moved"},
            {"role": "object", "quote": "the crate"},
        ],
    }
    with pytest.raises(ValueError, match="quote"):
        compile_frame(response, item, units)


def test_alias_exposed_by_another_role_is_rejected_without_losing_other_masks():
    response = "The crane was inspected. It lifted its load."
    units = {u["id"]: u for u in response_units(response)}
    item = {
        "unit_id": "u1",
        "anchor_quote": "It lifted its load.",
        "roles": [
            {"role": "subject", "quote": "It", "referent_quote": "The crane"},
            {"role": "predicate", "quote": "lifted"},
            {"role": "object", "quote": "its load", "referent_quote": "The crane"},
        ],
    }
    compiled = compile_frame(response, item, units)
    assert {r["target_role"] for r in compiled["rejected_targets"]} == {
        "subject",
        "object",
    }
    assert [q["atomic_target_role"] for q in compiled["questions"]] == ["predicate"]
    predicate = compiled["questions"][0]
    assert predicate["question_struct"] == json.loads(predicate["question"])
    assert predicate["hidden_answer_aliases"] == ["lifted"]
    assert len(predicate["decontext_bindings"]) == 2


def condition_prediction():
    return {
        "answerability": "not_stated",
        "all_conditions_covered": True,
        "condition_checks": [
            {
                "condition_role": "subject",
                "condition_quote": "The crane",
                "status": "supported",
            },
            {
                "condition_role": "predicate",
                "condition_quote": "moved",
                "status": "supported",
            },
        ],
    }


def test_condition_binding_requires_every_role_and_preserves_raw_prediction():
    question = json.dumps(
        {
            "requested_role": "object",
            "conditions": {"subject": "The crane", "predicate": "moved"},
        }
    )
    prediction = condition_prediction()
    valid = bind_condition_checks(prediction, question)
    assert valid["atomic_condition_binding"]["valid"]
    assert valid["answerability"] == "not_stated"
    prediction["condition_checks"].pop()
    invalid = bind_condition_checks(prediction, question)
    assert invalid["answerability"] == "uncertain"
    assert invalid["raw_answerability"] == "not_stated"
    assert prediction["answerability"] == "not_stated"
    assert invalid["all_conditions_covered"] is False


@pytest.mark.parametrize(
    "change", ["duplicate", "extra", "wrong_value", "missing_premise"]
)
def test_no_target_absence_from_wrong_or_unresolved_condition_binding(change):
    question = json.dumps(
        {
            "requested_role": "object",
            "conditions": {"subject": "The crane", "predicate": "moved"},
        }
    )
    prediction = condition_prediction()
    checks = prediction["condition_checks"]
    if change == "duplicate":
        checks.append(dict(checks[0]))
    elif change == "extra":
        checks.append(
            {
                "condition_role": "object",
                "condition_quote": "the crate",
                "status": "supported",
            }
        )
    elif change == "wrong_value":
        checks[0]["condition_quote"] = "the truck"
    else:
        checks[0]["status"] = "missing"
    assert bind_condition_checks(prediction, question)["answerability"] == "uncertain"


def test_frame_call_budget_keeps_unprocessed_units_explicit():
    from route_graph.atomic_anchor import extract_atomic_questions
    from route_graph.audit_protocol import PROTOCOL

    class Reader:
        def __init__(self):
            self.calls = []

        def ask(self, instruction, payload, **kwargs):
            self.calls.append((payload, kwargs))
            return {"frames": [], "nonassertion_unit_ids": []}

    reader = Reader()
    config = {
        **PROTOCOL,
        "atomic_units_per_batch": 1,
        "atomic_batches_per_response": 1,
        "atomic_frames_per_batch": 2,
        "atomic_frame_json_token_limit": 321,
    }
    result = extract_atomic_questions(
        reader, "The crane moved. The truck stopped.", protocol=config
    )
    assert len(reader.calls) == 1
    assert reader.calls[0][0]["frame_limit"] == 2
    assert reader.calls[0][1]["max_new_tokens"] == 321
    assert result["atomic_budget"]["unprocessed_unit_ids"] == ["u1"]
    assert result["atomic_budget"]["status"] == "exhausted"


def test_frame_budget_counts_rejected_frames_and_enforces_protocol_role_bound():
    from route_graph.atomic_anchor import extract_atomic_questions
    from route_graph.audit_protocol import PROTOCOL

    class Reader:
        def ask(self, instruction, payload, **kwargs):
            return {"frames": [frame(), frame(), frame()]}

    config = {**PROTOCOL, "atomic_frames_per_response": 1, "atomic_role_word_limit": 1}
    result = extract_atomic_questions(
        Reader(), frame()["anchor_quote"], protocol=config
    )
    assert result["atomic_budget"]["frames_considered"] == 1
    assert result["atomic_budget"]["returned_frames_omitted_by_budget"] == 2
    assert not result["questions"]
    assert (
        result["atomic_frame_failures"][0]["reason"]
        == "role is longer than the frozen constituent bound"
    )
