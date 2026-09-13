import copy

import pytest

from route_graph.source_event_graph import (
    compile_literal_fields,
    compile_pointer_events,
    raw_inventory,
    validate_inventory,
    validate_pointer_graph,
)


def pointer(a, b, unit="u0"):
    return {"unit_id": unit, "start_token": a, "end_token": b}


def envelope(inventory, prediction):
    return {"inventory_sha256": inventory["sha256"], "side": inventory["side"],
            "sample_id": inventory["sample_id"], "prediction": prediction}


def event():
    return {"anchor": pointer(0, 5), "roles": [
        {"role": "subject", "pointer": pointer(0, 2)},
        {"role": "predicate", "pointer": pointer(2, 3)},
        {"role": "object", "pointer": pointer(3, 4)},
    ]}


def test_inventory_retains_all_characters_but_does_not_claim_fact_coverage():
    text = "  α crane moved itself.\n\nNone? \t"
    inv = raw_inventory(text, side="source", sample_id="s", unit_tokens=3)
    assert "".join(text[slice(*u["span"])] for u in inv["units"]) == text
    assert inv["coverage_kind"] == "raw_characters_not_facts"
    validate_inventory(inv)


def test_event_citations_are_derived_from_pointers_and_independent_sides():
    text = "The crane observed itself."
    source = raw_inventory(text, side="source", sample_id="s")
    response = raw_inventory(text, side="response", sample_id="r")
    raw = {"events": [event()]}
    graph = compile_pointer_events(source, envelope(source, raw))
    other = compile_pointer_events(response, envelope(response, raw))
    assert graph["nodes"][0]["quote"] == "The crane"
    assert graph["nodes"][0]["id"] != other["nodes"][0]["id"]
    assert graph["absence_inference_allowed"] is False
    raw["events"][0]["roles"][0]["pointer"]["start_token"] = 1
    validate_pointer_graph(source, graph)
    with pytest.raises(ValueError, match="another inventory"):
        validate_pointer_graph(response, graph)


def test_intransitive_two_role_event_is_retained_and_partial_remains_partial():
    inv = raw_inventory("Stocks fell.", side="source", sample_id="s")
    raw = {"anchor": pointer(0, 3), "roles": [
        {"role": "subject", "pointer": pointer(0, 1)},
        {"role": "predicate", "pointer": pointer(1, 2)},
    ]}
    complete = compile_pointer_events(inv, envelope(inv, {"events": [raw]}))
    assert complete["events"][0]["status"] == "parsed_event"
    raw["roles"].pop(0)
    partial = compile_pointer_events(inv, envelope(inv, {"events": [raw]}))
    assert partial["events"][0]["status"] == "partial_event"
    assert partial["events"][0]["relation_verification"] == "not_performed"


@pytest.mark.parametrize("change", ["free_quote", "out_of_bounds", "boolean", "overlap", "duplicate", "reversed"])
def test_bad_event_references_cannot_create_retained_nodes(change):
    inv = raw_inventory("The crane observed itself.", side="source", sample_id="s")
    raw = event()
    if change == "free_quote":
        raw["source_quote"] = "invented"
    elif change == "out_of_bounds":
        raw["roles"][0]["pointer"]["end_token"] = 999
    elif change == "boolean":
        raw["roles"][0]["pointer"]["start_token"] = False
    elif change == "overlap":
        raw["roles"][1]["pointer"] = pointer(1, 3)
    elif change == "duplicate":
        raw["roles"].append(copy.deepcopy(raw["roles"][0]))
    else:
        raw["roles"].reverse()
    graph = compile_pointer_events(inv, envelope(inv, {"events": [raw]}))
    assert not graph["events"] and not graph["nodes"]
    assert len(graph["failures"]) == 1
    assert graph["coverage"]["raw_tokens"] == 5
    assert graph["coverage"]["event_anchor_tokens"] == 0


def test_literal_structure_preserves_unknown_unicode_coordinates_and_field_provenance():
    text = "{'名字': 'café',\n 'hours': {'Monday': None}, 'rating': -3.5, 'reviews': [True, 'ok']}"
    inv = raw_inventory(text, side="source", sample_id="s")
    graph = compile_literal_fields(inv)
    validate_pointer_graph(inv, graph)
    assert all(text[slice(*n["span"])] == n["raw_literal"] for n in graph["nodes"])
    unknown = next(n for n in graph["nodes"] if n["path"] == ["hours", "Monday"])
    assert unknown["epistemic_status"] == "unknown" and unknown["value"] is None
    assert next(n for n in graph["nodes"] if n["path"] == ["rating"])["value"] == -3.5
    assert graph["relation_verification"] == "not_performed"
    graph["nodes"][0]["raw_literal"] = "another citation"
    with pytest.raises(ValueError, match="differs"):
        validate_pointer_graph(inv, graph)


@pytest.mark.parametrize("text", [
    "{'x': dangerous()}", "{'x': object.attribute}", "{'x': 2 + 3}",
    "{**other}", "{'x': [n for n in values]}", "{'x': 1, 'x': 2}",
    "{'x': 1e309}", "{'x': -True}", "{'x': (1, 2)}",
])
def test_nonliteral_and_ambiguous_sources_are_rejected_without_execution(text):
    inv = raw_inventory(text, side="source", sample_id="s")
    with pytest.raises(ValueError):
        compile_literal_fields(inv)


def test_inventory_coordinate_tampering_is_rejected():
    inv = raw_inventory("The crane observed itself.", side="source", sample_id="s")
    inv["units"][0]["tokens"][0]["span"] = [4, 9]
    with pytest.raises(ValueError, match="integrity"):
        compile_pointer_events(inv, envelope(inv, {"events": [event()]}))


def test_foreign_source_request_cannot_be_rebound_to_same_shaped_inventory():
    a = raw_inventory("The crane observed itself.", side="source", sample_id="a")
    b = raw_inventory("The pilot observed herself.", side="source", sample_id="b")
    with pytest.raises(ValueError, match="another inventory"):
        compile_pointer_events(b, envelope(a, {"events": [event()]}))


@pytest.mark.parametrize("raw", [None, [], "malformed model response", {"reader_error": "invalid_json"}, {"events": "bad list"}])
def test_malformed_model_roots_remain_failure_graphs_with_full_raw_denominator(raw):
    inv = raw_inventory("The crane observed itself.", side="source", sample_id="s")
    graph = compile_pointer_events(inv, envelope(inv, raw))
    assert graph["coverage"]["raw_tokens"] == 5
    assert graph["coverage"]["event_anchor_tokens"] == 0
    assert graph["failures"][0]["scope"] == "root"
    assert graph["prediction_envelope"]["prediction"] == raw
    assert not graph["nodes"] and not graph["events"]
    validate_pointer_graph(inv, graph)
