"""Review tests for complete constraint inventory provenance boundaries."""

from next_iteration.constraint_inventory import compile_inventory, decoded_string_map


def _all_inventory_node_refs(inventory):
    nodes = inventory["fields"] + inventory["contexts"] + inventory["components"] + inventory["bundles"]
    return {node["id"] for node in nodes}


def test_decoded_string_map_preserves_escape_raw_spans():
    raw = r"'caf\u00e9 \141 \' \\ end'"
    expected = "café a ' \\ end"

    spans = decoded_string_map(raw, 10, expected)

    assert len(spans) == len(expected)
    assert raw[slice(*(span - 10 for span in spans[3]))] == r"\u00e9"
    assert raw[slice(*(span - 10 for span in spans[5]))] == r"\141"
    assert raw[slice(*(span - 10 for span in spans[7]))] == r"\'"
    assert raw[slice(*(span - 10 for span in spans[9]))] == r"\\"


def test_data2txt_inventory_keeps_unknown_long_weekday_and_address_components():
    long_review = " ".join(f"token{i}" for i in range(45))
    source = repr({
        "name": "Cafe",
        "address": "220 1st Ave",
        "attributes": {"WiFi": None, "Music": None},
        "hours": {day: "8:0-16:0" for day in (
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        )},
        "review_info": [{"review_text": long_review, "review_stars": 4.0}],
    })

    inventory = compile_inventory(source, source_id="9001", task="Data2txt")

    assert inventory["labels_read"] is False
    assert inventory["model_forwards"] == 0
    assert inventory["semantic_certificate_count"] == 0
    fields_by_path = {tuple(field["field_path"]): field for field in inventory["fields"]}
    assert fields_by_path[("attributes", "WiFi")]["value_status"] == "unknown"
    assert fields_by_path[("attributes", "Music")]["value_status"] == "unknown"
    assert fields_by_path[("review_info", 0, "review_text")]["display_text"] == long_review
    assert fields_by_path[("review_info", 0, "review_text")]["component_mapping_status"] == "exact"

    address_field = fields_by_path[("address",)]
    address_components = [
        c for c in inventory["components"] if c["parent_id"] == address_field["id"]
    ]
    assert any(c["display_text"] == "220" and c["surface_type"] == "number" for c in address_components)

    weekday_bundles = [b for b in inventory["bundles"] if b["kind"] == "weekday_key_inventory"]
    assert len(weekday_bundles) == 1
    assert weekday_bundles[0]["source_key_count"] == 7
    assert weekday_bundles[0]["query_requirement_verified"] is False
    assert weekday_bundles[0]["semantic_joint_fact"] is False

    node_ids = _all_inventory_node_refs(inventory)
    for edge in inventory["edges"]:
        assert edge["from"] in node_ids
        assert edge["to"] in node_ids
    for node in inventory["fields"] + inventory["contexts"] + inventory["components"]:
        assert source[slice(*node["raw_span"])] == node["raw_surface"]


def test_natural_source_inventory_keeps_full_document_owner_and_component_refs():
    source = "Passage 1 says the cafe is open daily from 8 AM to 4 PM, but WiFi is not stated."

    inventory = compile_inventory(source, source_id="42", task="QA")

    assert inventory["source_kind"] == "source_surface"
    assert inventory["fields"] == [{
        "id": inventory["fields"][0]["id"],
        "kind": "source_document_owner",
        "raw_span": [0, len(source)],
        "display_text": source,
        "raw_surface": source,
        "value_status": "observed_text_not_fact_truth",
        "owner_available": True,
        "field_path": [],
        "record_id": None,
        "factual_status": "unverified",
        "component_mapping_status": "exact",
    }]
    assert inventory["components"]
    assert inventory["bundles"]
    node_ids = _all_inventory_node_refs(inventory)
    assert all(edge["from"] in node_ids and edge["to"] in node_ids for edge in inventory["edges"])
    assert all(source[slice(*node["raw_span"])] == node["raw_surface"] for node in inventory["contexts"] + inventory["components"])
    assert all(bundle["semantic_joint_fact"] is False for bundle in inventory["bundles"])


def test_unsupported_concatenated_string_keeps_whole_field_without_fake_components():
    source = "{'name': 'ab', 'note': 'a' 'b'}"

    inventory = compile_inventory(source, source_id="9002", task="Data2txt")

    fields_by_path = {tuple(field["field_path"]): field for field in inventory["fields"]}
    note = fields_by_path[("note",)]
    assert note["display_text"] == "ab"
    assert note["component_mapping_status"] == "unavailable_preserve_whole_field"
    assert any(failure["field_id"] == note["id"] for failure in inventory["mapping_failures"])
    assert source[slice(*note["raw_span"])] == note["raw_surface"]
    assert not [component for component in inventory["components"] if component["parent_id"] == note["id"]]
