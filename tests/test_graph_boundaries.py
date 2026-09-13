"""Regression of observed boundary/scope failures, without factual labels."""

import pytest

from next_iteration.graph_boundaries import (
    base_graph,
    graph_regions,
    localized_words,
    sentence_envelopes,
)
from route_graph.source_event_graph import compile_pointer_events, raw_inventory


def graph_for(text, *, window=128, event=None):
    inventory = raw_inventory(text, side="response", sample_id="fixture", unit_tokens=window)
    pointer = compile_pointer_events(inventory, {
        "inventory_sha256": inventory["sha256"], "sample_id": "fixture", "side": "response",
        "prediction": {"events": [] if event is None else [event]}})
    assert not pointer["failures"]
    return base_graph(inventory, pointer)


def point(a, b):
    return {"unit_id": "u0", "start_token": a, "end_token": b}


def test_address_windows_cannot_split_a_sentence_or_noun_coordination():
    text = "The cardiovascular system regulates flow between the digestive and respiratory systems."
    short = graph_for(text, window=8)
    large = graph_for(text, window=128)
    assert [b["span"] for b in short["base_units"]] == [b["span"] for b in large["base_units"]] == [[0, len(text)]]
    assert short["base_units"][0]["address_boundaries_inside"]
    very_long = graph_for("A " + "detailed " * 260 + "description.")
    assert len(very_long["base_units"]) == 1


def test_coordinate_valid_incomplete_event_never_truncates_readable_context():
    text = "Archaeologists have discovered a stone chest in a 1,350-year-old church in Turkey."
    event = {"anchor": point(0, 8), "roles": [
        {"role": "subject", "pointer": point(0, 1)},
        {"role": "predicate", "pointer": point(1, 3)},
        {"role": "object", "pointer": point(3, 6)}]}
    graph = graph_for(text, event=event)
    assert len(graph["base_units"]) == 1
    assert graph["base_units"][0]["text"] == text
    assert graph["nested_events"][0]["status"] == "truncated_anchor_candidate"


@pytest.mark.parametrize(("text", "count"), [
    ('She called them "very kind." She left.', 2),
    ('1. First result.\n2. Second result.', 2),
    ('Dr. Smith counted 3.14 units. He left.', 2),
    ('"Done," she said. A new sentence.', 2),
    ('Alpha won in 2019, and Beta won in 2020.', 1),
    ('A title\nwrapped onto another line.\n\nNext paragraph.', 2),
])
def test_punctuation_lists_and_wrapped_lines_keep_intact_words(text, count):
    graph = graph_for(text)
    assert len(graph["base_units"]) == count
    spans = sentence_envelopes(text)
    assert "".join(text[a:b] for a, b in spans) == text
    words = localized_words(text, graph, {b["base_id"]: .99 for b in graph["base_units"]}, [])
    assert len(words) == len(text.split())
    assert all(w["localized_risk"] == .5 for w in words)


def quantity_graph():
    text = "Alpha shipped 9 crates."
    event = {"anchor": point(0, 5), "roles": [
        {"role": "subject", "pointer": point(0, 1)},
        {"role": "predicate", "pointer": point(1, 2)},
        {"role": "quantity", "pointer": point(2, 3)}]}
    graph = graph_for(text, event=event)
    role = next(r for r in graph["nested_roles"] if r["role"] == "quantity")
    decision = {"decision_id": "d", "base_id": role["base_id"], "role_id": role["role_id"],
                "target_span": role["span"], "risk": .95, "scope_verified": True,
                "basis": "validated_single_slot_bridge", "evidence_record_sha256": "a" * 64}
    return text, graph, decision


def test_one_wrong_slot_does_not_spread_bag_risk_to_other_sentence_words():
    text, graph, decision = quantity_graph()
    words = localized_words(text, graph, {decision["base_id"]: .99}, [decision])
    assert [w["text"] for w in words if w["localized_risk"] > .5] == ["9"]
    assert all(w["base_bag_risk"] == .99 for w in words)
    decision["scope_verified"] = False
    assert all(w["localized_risk"] == .5 for w in localized_words(text, graph, {decision["base_id"]: .99}, [decision]))


def test_opposed_local_decisions_and_misbound_slots_stay_unresolved():
    text, graph, decision = quantity_graph()
    other = {**decision, "decision_id": "other", "risk": .05}
    words = localized_words(text, graph, {decision["base_id"]: .99}, [decision, other])
    assert next(w for w in words if w["text"] == "9")["localization_status"] == "conflicting_local_relations"
    with pytest.raises(ValueError, match="target differs"):
        localized_words(text, graph, {decision["base_id"]: .99}, [{**decision, "target_span": [0, len(text)]}])


def test_nonadjacent_reuse_preserves_edges_without_marking_middle_text_wrong():
    graph = graph_for("The first assertion. An unrelated remark. Reuse of the first assertion.")
    ids = [b["base_id"] for b in graph["base_units"]]
    edge = {"edge_id": "e", "from_base": ids[0], "to_base": ids[2], "kind": "reuse",
            "probability": .9, "specific_fact_link": True}
    regions = graph_regions(graph, [edge])
    linked = next(r for r in regions["connected_regions"] if len(r["base_ids"]) == 2)
    assert linked["base_ids"] == [ids[0], ids[2]] and linked["noncontiguous"]
    assert all(len(r["base_ids"]) == 1 for r in regions["contiguous_regions"])
    assert all(r["assigns_word_risk"] is False for r in regions["connected_regions"])


def test_correction_blocks_an_adjacent_merge_even_with_shared_topic():
    graph = graph_for("The first assertion. A correction.")
    ids = [b["base_id"] for b in graph["base_units"]]
    edge = {"edge_id": "e", "from_base": ids[0], "to_base": ids[1], "kind": "reuse",
            "probability": .9, "specific_fact_link": True}
    correction = {**edge, "edge_id": "c", "kind": "correction"}
    result = graph_regions(graph, [edge, correction])
    assert len(result["contiguous_regions"]) == 2 and result["conflicting_pairs"]


def test_localization_rejects_changed_text_at_frozen_role_coordinates():
    text, graph, decision = quantity_graph()
    with pytest.raises(ValueError, match="text"):
        localized_words(text.replace("9", "7"), graph, {decision["base_id"]: .99}, [decision])


def test_quote_only_base_remains_explicitly_nonassertive_at_word_output():
    text = '""'
    graph = graph_for(text)
    words = localized_words(text, graph, {graph["base_units"][0]["base_id"]: .99}, [])
    assert words[0]["localization_status"] == "nonassertive"


def test_jump_edge_keeps_a_connected_region_noncontiguous_even_when_ids_fill_the_gap():
    graph = graph_for("First fact. Middle text. Third fact.")
    ids = [b["base_id"] for b in graph["base_units"]]
    edges = [
        {"edge_id": "jump", "from_base": ids[0], "to_base": ids[2], "kind": "reuse",
         "probability": .9, "specific_fact_link": True},
        {"edge_id": "adjacent", "from_base": ids[1], "to_base": ids[2], "kind": "reuse",
         "probability": .9, "specific_fact_link": True},
    ]
    region = graph_regions(graph, edges)["connected_regions"][0]
    assert region["noncontiguous"] is True
