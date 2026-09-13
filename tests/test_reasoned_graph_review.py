import json
import re

from next_iteration.constraint_inventory import compile_inventory
from next_iteration.reasoned_graph import PROTOCOL, compile_prediction, packet, ranges, source_view
from route_graph.frozen_reader import digest


def _inventory():
    source = repr({
        "name": "Cafe",
        "address": "220 State St",
        "business_stars": 4.0,
        "attributes": {"WiFi": None},
        "hours": {"Monday": "8:0-16:0", "Tuesday": "8:0-16:0"},
    })
    return compile_inventory(source, source_id="123", task="Data2txt")


def _row(inventory):
    prefix = "SOURCE:\n"
    suffix = "\nRESPONSE:\n"
    prompt = prefix + inventory["text"] + suffix
    return {
        "id": "r1",
        "source_id": inventory["source_id"],
        "task": inventory["task"],
        "prompt": prompt,
        "source_span": [len(prefix), len(prefix) + len(inventory["text"])],
        "response": "The cafe has 4 stars and WiFi.",
        "response_sha256": digest("The cafe has 4 stars and WiFi."),
    }


def _request():
    inv = _inventory()
    return packet(_row(inv), inv, "graph")


def test_source_view_retains_all_raw_characters_with_disjoint_inline_intervals():
    inventory = _inventory()
    view = source_view(inventory)

    assert re.sub(r"</?s\d+>", "", view["text"]) == inventory["text"]
    assert view["all_source_characters_retained"] is True

    spans = [node["raw_span"] for node in view["nodes"]]
    assert spans == sorted(spans)
    previous = 0
    for start, end in spans:
        assert previous <= start < end <= len(inventory["text"])
        previous = end


def test_packet_flat_and_graph_share_identical_source_and_response_markers():
    inventory = _inventory()
    row = _row(inventory)
    flat = packet(row, inventory, "flat")
    graph = packet(row, inventory, "graph")

    assert flat["payload"]["SOURCE"] == graph["payload"]["SOURCE"]
    assert flat["payload"]["RESPONSE"] == graph["payload"]["RESPONSE"]
    assert flat["payload"]["PROMPT_BEFORE_SOURCE"] == row["prompt"][: row["source_span"][0]]
    assert flat["payload"]["PROMPT_AFTER_SOURCE"] == row["prompt"][row["source_span"][1] :]
    assert flat["payload"]["word_count"] == graph["payload"]["word_count"] == len(graph["words"])
    assert "source_topology" not in flat["payload"]
    assert graph["payload"]["source_topology"]["semantics"].startswith("observed source provenance")


def test_graph_topology_honestly_declares_coarse_scope_without_component_pointers():
    inventory = _inventory()
    graph = packet(_row(inventory), inventory, "graph")
    topology = graph["payload"]["source_topology"]

    assert PROTOCOL["role"] == "external_semantic_reference; not internal-only ownership method"
    assert PROTOCOL["rendered_topology_scope"] == (
        "coarse field/record and text-context provenance; full component graph stays in inventory"
    )
    assert graph["source"]["inventory_counts"]["components"] == len(inventory["components"])
    assert len(inventory["components"]) > 0
    assert len(topology["nodes"]) == len(inventory["fields"])
    assert all(node["id"].startswith("s") for node in topology["nodes"])
    exposed_bundle_members = {member for bundle in topology["bundles"] for member in bundle["members"]}
    assert all(member.startswith("s") for member in exposed_bundle_members)
    assert all(bundle["observed_containment_only"] for bundle in topology["bundles"])
    assert all(bundle["query_requirement_verified"] is False for bundle in topology["bundles"])


def test_compile_prediction_retains_invalid_evidence_and_bad_repair_without_losing_denominator():
    request = _request()
    raw = json.dumps({
        "facts": [{
            "id": "f0",
            "members": [[0, 5]],
            "target": [[3, 5]],
            "p": [0.1, 0.7, 0.1, 0.1],
            "evidence": ["s0", "missing-source"],
            "applicability": "model predicts a rating conflict",
            "repair": {"range": [0, 1], "replacement": "A", "preservation": "predicted_preserved"},
        }],
        "history": [],
        "nonfactual": [],
    })

    result = compile_prediction(request, raw)

    assert len(result["facts"]) == 1
    assert result["facts"][0]["pointer_status"] == "invalid"
    assert result["facts"][0]["repair_candidate"] is None
    assert any(f["kind"] == "repair" for f in result["failures"])
    assert len(result["words"]) == len(request["words"])
    assert result["words"][3]["risk"] == 0.7 + 0.1 + 0.5 * 0.1
    assert result["words"][-1]["risk"] == 0.5


def test_compile_prediction_allows_overlapping_fact_targets_and_averages_word_risk():
    request = _request()
    raw = json.dumps({
        "facts": [
            {"id": "f0", "members": [[0, 5]], "target": [[3, 5]], "p": [0.8, 0.1, 0.0, 0.1], "evidence": [], "applicability": "rating assertion", "repair": None},
            {"id": "f1", "members": [[3, 6]], "target": [[3, 4]], "p": [0.2, 0.4, 0.3, 0.1], "evidence": [], "applicability": "number assertion", "repair": None},
        ],
        "history": [],
        "nonfactual": [],
    })

    result = compile_prediction(request, raw)

    assert result["counts"]["valid_fact_coordinates_and_scores"] == 2
    assert result["words"][3]["fact_ids"] == ["f0", "f1"]
    assert result["words"][3]["risk"] == ((0.1 + 0.0 + 0.5 * 0.1) + (0.4 + 0.3 + 0.5 * 0.1)) / 2


def test_malformed_json_or_wrong_top_level_types_keep_complete_neutral_denominator():
    request = _request()

    for raw in ["[1, 2, 3]", json.dumps({"facts": {}, "history": [], "nonfactual": []})]:
        result = compile_prediction(request, raw)
        assert any(f["kind"] == "response_parse" for f in result["failures"])
        assert result["counts"]["words"] == len(request["words"])
        assert result["counts"]["valid_fact_coordinates_and_scores"] == 0
        assert all(word["risk"] == 0.5 and word["status"] == "uncovered" for word in result["words"])
        assert result["not_ground_truth"] is True
        assert result["labels_read"] is False


def test_ranges_reject_overlapping_ranges_inside_one_fact():
    request = _request()
    try:
        ranges([[1, 3], [2, 4]], request["words"])
    except ValueError as error:
        assert "ordered, disjoint" in str(error)
    else:
        raise AssertionError("overlapping ranges inside one fact should be rejected")
