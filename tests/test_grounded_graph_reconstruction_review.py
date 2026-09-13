"""Mechanical provenance checks for deterministic source-coordinate reconstruction."""

import json
import re

import pytest

from next_iteration import grounded_graph_reconstruction as reconstruction_module
from next_iteration.constraint_inventory import compile_inventory
from next_iteration.grounded_graph_reconstruction import PROTOCOL, reconstruction
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest


def _word_spans(text):
    return [match.span() for match in re.finditer(r"\w+(?:['’\-]\w+)*", text)]


def test_data2txt_targets_are_observed_first_words_after_global_warmup_and_never_unknown():
    text = repr({
        "name": "A very long cafe name", "address": "10 West Road", "rating": 4.5,
        "attributes": {"WiFi": None}, "notes": "extra observed details for coordinate targets",
    })
    inventory = compile_inventory(text, source_id="3", task="Data2txt")
    result = reconstruction(inventory)
    owners = {field["id"]: field for field in inventory["fields"]}
    cutoff = _word_spans(text)[PROTOCOL["minimum_global_prefix_words"]][0]

    assert result["text"] == text and result["semantic_ground_truth"] is False
    assert result["targets"]
    for target in result["targets"]:
        owner = owners[target["source_owner_id"]]
        first = re.search(r"\w+(?:['’\-]\w+)*", text[slice(*owner["raw_span"])])
        assert owner["value_status"] != "unknown"
        assert target["target_span"] == target["source_span"]
        assert target["target_span"][0] >= cutoff
        assert text[slice(*target["target_span"])] == target["raw_value"]
        assert target["target_span"] == [owner["raw_span"][0] + first.start(), owner["raw_span"][0] + first.end()]
        assert target["supervision"] == "observed_copy_coordinate; semantic_owner_unverified"


def test_qa_context_targets_keep_exact_context_owner_coordinates_without_semantic_upgrade():
    text = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen."
    inventory = compile_inventory(text, source_id="4", task="QA")
    result = reconstruction(inventory)
    contexts = {context["id"]: context for context in inventory["contexts"]}

    assert result["targets"]
    for target in result["targets"]:
        context = contexts[target["source_owner_id"]]
        assert context["raw_span"][0] <= target["target_span"][0] < target["target_span"][1] <= context["raw_span"][1]
        assert text[slice(*target["target_span"])] == target["raw_value"]
        assert result["semantic_ground_truth"] is False


def test_verifier_rejects_a_self_consistent_manifest_that_omits_selected_source(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text("frozen source roster\n")
    inventory = tmp_path / "inventory"
    (inventory / "sources").mkdir(parents=True)
    (inventory / "manifest.json").write_text("inventory\n")
    for source_id in ("a", "b"):
        (inventory / "sources" / f"{source_id}.json").write_text("sealed source\n")
    inventory_settings = {"sealed": "settings"}
    settings = {
        "protocol": PROTOCOL,
        "code_sha256": {},
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "inventory_path": str(inventory),
        "inventory_manifest_sha256": file_sha256(inventory / "manifest.json"),
        "inventory_settings_sha256": "inventory-settings",
        "inventory_settings_digest": digest(inventory_settings),
        "source_artifacts": {f"sources/{source_id}.json": file_sha256(inventory / "sources" / f"{source_id}.json")
                             for source_id in ("a", "b")},
        "selection_census": {"Data2txt:train": {"selected": 2}},
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    # The listed source artifact set says a/b were selected, but examples retains only a.
    (tmp_path / "examples.json").write_text(json.dumps([{"source_id": "a"}]))
    (tmp_path / "summary.json").write_text(json.dumps({"sources": 1}))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "settings_sha256": file_sha256(tmp_path / "settings.json"),
        "artifacts": {name: file_sha256(tmp_path / name) for name in ("examples.json", "summary.json")},
    }))
    monkeypatch.setattr(
        reconstruction_module,
        "inventory_parent",
        lambda *_: ({"settings_file_sha256": "inventory-settings"}, inventory_settings),
    )

    with pytest.raises(ValueError, match="omits or duplicates selected source IDs"):
        reconstruction_module.verify(tmp_path)
