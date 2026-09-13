import copy

import pytest

from next_iteration.fact_scope_mask import masked_view, owner_payload, validate_view


def test_coordinate_mask_preserves_homograph_in_condition_and_name():
    text = "Shop 1 is open on day 1 until 1."
    start = text.rindex("1")
    masks = [{"id": "possibly_value_derived_1", "span": [start, start + 1], "role": "target_value"}]
    view = masked_view(text, document_id="1", masks=masks, protected=[[0, start]])
    assert view["text"] == "Shop 1 is open on day 1 until <VALUE_0>."
    payload = owner_payload(view, [])
    assert "possibly_value_derived" not in str(payload)
    assert payload["response_context"].count("1") == 2
    validate_view(text, view, masks=masks, protected=[[0, start]])


def test_same_surface_distinct_source_records_keep_distinct_masks():
    text = "A: 4.0; B: 4.0"
    view = masked_view(text, document_id="source", masks=[
        {"id": "A", "span": [3, 6], "role": "candidate_value"},
        {"id": "B", "span": [11, 14], "role": "candidate_value"}])
    assert view["text"] == "A: <VALUE_0>; B: <VALUE_1>"
    assert view["mask_count"] == 2


def test_mask_cannot_delete_required_constraint():
    with pytest.raises(ValueError, match="non-target constraint"):
        masked_view("8 every day", document_id="r", masks=[
            {"id": "x", "span": [0, 11], "role": "target_value"}], protected=[[2, 11]])


def test_linked_repetition_needs_explicit_mask():
    view = masked_view("8 then 8", document_id="r", masks=[
        {"id": "t", "span": [7, 8], "role": "target_value"},
        {"id": "linked", "span": [0, 1], "role": "linked_value"}])
    assert view["text"] == "<VALUE_0> then <VALUE_1>"
    assert view["dependency_completeness"] == "caller_obligation_not_inferred_from_text"


def test_tampered_view_is_rejected_at_payload_boundary():
    view = masked_view("8", document_id="r", masks=[{"id": "t", "span": [0, 1], "role": "target_value"}])
    changed = copy.deepcopy(view)
    changed["text"] = "8"
    with pytest.raises(ValueError):
        owner_payload(changed, [])


def test_owner_payload_requires_actual_response_target_mask():
    view = masked_view("unmasked", document_id="r", masks=[])
    with pytest.raises(ValueError, match="response target"):
        owner_payload(view, [])


def test_owner_payload_rejects_unmasked_source_value_view():
    response = masked_view("8", document_id="r", masks=[{"id": "t", "span": [0, 1], "role": "target_value"}])
    source = masked_view("source value: 8", document_id="s", masks=[])
    with pytest.raises(ValueError, match="source view"):
        owner_payload(response, [source])
