import copy

import pytest

from next_iteration.surface_graph import (
    edit_candidate,
    source_occurrences,
    surface_graph,
)


def response(text):
    return surface_graph(text, sample_id="r", side="response")


def test_complete_base_crosses_old_address_and_needs_no_srl():
    text = "Archaeologists have discovered a stone chest in a 1,350-year-old church in Israel."
    graph = response(text)
    assert len(graph["base_units"]) == 1
    assert "1,350-year-old" in {s["quote"] for s in graph["slots"]}
    assert graph["semantic_role_extraction_required"] is False
    long = response("The report described " + "large " * 160 + "differences between the digestive and respiratory systems.")
    assert len(long["base_units"]) == 1


def test_contraction_not_fragmented_and_no_function_word_slot():
    graph = response("She doesn't miss her co-hosts, but instead misses the wardrobe crew.")
    quotes = {s["quote"] for s in graph["slots"]}
    assert "doesn" not in quotes and "doesn't" not in quotes and "on" not in quotes
    assert "wardrobe crew" in quotes


def test_maximal_lexical_surface_keeps_oil():
    slots = response("Wintergreen essential oil can soothe muscles.")["slots"]
    assert "Wintergreen essential oil" in {s["quote"] for s in slots}
    assert "Wintergreen essential" not in {s["quote"] for s in slots}
    assert all(s["not_semantic_role"] for s in slots)


def test_time_range_not_split_into_four_values():
    slots = response("The hours are from 8 AM to 6 PM every day.")["slots"]
    assert [(s["quote"], s["surface_type"]) for s in slots if s["surface_type"] == "time_range"] == [("8 AM to 6 PM", "time_range")]
    assert not any(s["quote"] in {"8", "6", "AM", "PM"} for s in slots)


def test_numeric_range_not_mislabeled_time():
    slots = response("The range is 8-16.")["slots"]
    assert any(s["quote"] == "8-16" and s["surface_type"] == "number_range" for s in slots)
    assert not any(s["surface_type"] == "time_range" for s in slots)


def test_literal_short_minutes_and_named_ampersand_are_whole_surfaces():
    source = source_occurrences("{'hours': {'Tuesday': '8:0-16:0'}, 'name': 'Crushcakes & Cafe'}", sample_id="s", task="Data2txt")
    hours = next(o for o in source["occurrences"] if o["display_surface"] == "8:0-16:0")
    assert hours["surface_type"] == "time_range" and "Tuesday" in hours["masked_context"]
    slots = response("Crushcakes & Cafe is rated 4.0.")["slots"]
    assert "Crushcakes & Cafe" in {s["quote"] for s in slots}
    assert not any(s["quote"] in {"Crushcakes", "Cafe"} for s in slots)


def test_decimal_sentence_end_and_numbered_marker():
    slots = response("1. The cafe is rated 4.0.")["slots"]
    assert any(s["quote"] == "4.0" for s in slots)
    assert not any(s["quote"] == "1" for s in slots)


def test_trailing_citation_and_next_list_marker_do_not_become_fact_base():
    graph = response("1. It may reduce pain. (Passage 1)\n2. It can soothe muscles.")
    assert [b["text"] for b in graph["base_units"]] == ["1. It may reduce pain. (Passage 1)", "\n2. It can soothe muscles."]
    assert not any(s["quote"] in {"1", "2", "Passage"} for s in graph["slots"])


def test_same_value_different_records_retained_with_context():
    source = "[{'name': 'A', 'stars': 4.0}, {'name': 'B', 'stars': 4.0}]"
    occurrences = source_occurrences(source, sample_id="s", task="Data2txt")["occurrences"]
    values = [o for o in occurrences if o["display_surface"] == "4.0"]
    assert len(values) == 2 and values[0]["parent_id"] != values[1]["parent_id"]
    assert "'A'" in values[0]["masked_context"] and "'B'" in values[1]["masked_context"]
    assert all("4.0" not in o["masked_context"] for o in values)


def test_value_masking_does_not_depend_on_value():
    a = response("The cafe is rated 1.0.")
    b = response("The cafe is rated 9.999.")
    sa = next(s for s in a["slots"] if s["surface_type"] == "number")
    sb = next(s for s in b["slots"] if s["surface_type"] == "number")
    assert sa["masked_context"] == sb["masked_context"]


def test_single_edit_preserves_other_constraints_but_not_claimed_true():
    graph = response("The cafe is rated 5.0 and opens every day.")
    sources = source_occurrences("{'stars': 4.0}", sample_id="s", task="Data2txt")
    slot = next(s for s in graph["slots"] if s["quote"] == "5.0")
    draft = edit_candidate(graph, sources, slot["slot_id"], sources["occurrences"][0]["occurrence_id"])
    assert draft["edited_base"] == "The cafe is rated 4.0 and opens every day."
    assert draft["status"] == "candidate_not_semantically_verified"
    assert draft["certificate_count"] == 0


@pytest.mark.parametrize("literal,status", [("None", "unknown_null"), ("True", "boolean_verbalization_closed")])
def test_null_bool_are_not_verbalized(literal, status):
    sources = source_occurrences("{'open': " + literal + "}", sample_id="s", task="Data2txt")
    assert sources["occurrences"][0]["edit_policy"] == status


def test_quote_and_long_review_edit_closed():
    graph = response('The reviewer said "very nice place".')
    assert next(s for s in graph["slots"] if s["surface_type"] == "explicit_quote")["edit_policy"] == "quoted_edit_closed"
    sources = source_occurrences(repr({"review": "good " * 20}), sample_id="s", task="Data2txt")
    assert sources["occurrences"][0]["edit_policy"] == "long_or_multiline_literal_edit_closed"


def test_source_graph_tampering_rejected():
    graph = response("The cafe is rated 5.0.")
    sources = source_occurrences("{'stars': 4.0}", sample_id="s", task="Data2txt")
    bad = copy.deepcopy(sources)
    bad["occurrences"][0]["display_surface"] = "8.0"
    with pytest.raises(ValueError, match="content changed"):
        edit_candidate(graph, bad, graph["slots"][0]["slot_id"], bad["occurrences"][0]["occurrence_id"])
