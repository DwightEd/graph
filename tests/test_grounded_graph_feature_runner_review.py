"""Persisted feature-array contracts for the grounded graph runner."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from next_iteration import grounded_graph_feature_runner as feature_runner
from next_iteration.constraint_inventory import compile_inventory
from next_iteration.grounded_graph_feature_runner import load_example, model_identity
from next_iteration.grounded_graph_features import REPRESENTATION, prepare_example
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest


class CharacterTokenizer:
    bos_token_id = 1

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": [2 + (ord(character) % 89) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def _fixture(tmp_path):
    tokenizer = CharacterTokenizer()
    source = repr({"name": "Ada", "city": "Pisa"})
    inventory = compile_inventory(source, source_id="7", task="Data2txt")
    prompt = "Task:\n" + source + "\nAnswer:\n"
    start = prompt.index(source)
    prompt_ids = [1] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    owner = next(field for field in inventory["fields"] if field["display_text"] == "Ada")
    packet = prepare_example(
        {"source_id": "7", "prompt": prompt, "source_span": [start, start + len(source)],
         "prompt_token_ids": prompt_ids, "prompt_length": len(prompt_ids)},
        inventory,
        "The value is Ada.",
        [{"target_span": [13, 16], "source_owner_id": owner["id"]}],
        tokenizer,
    )
    inventory_path = tmp_path / "inventory"
    (inventory_path / "sources").mkdir(parents=True)
    (inventory_path / "sources" / "7.json").write_text(json.dumps({"data": inventory}))
    (tmp_path / "packets").mkdir()
    (tmp_path / "packets" / "e0.json").write_text(json.dumps(packet))
    (tmp_path / "features").mkdir()
    code_path = Path(__file__).parents[1] / "next_iteration" / "grounded_graph_features.py"
    settings = {
        "inventory_path": str(inventory_path),
        "model_files": [
            {"name": "weights.safetensors", "sha256": "model"},
            {"name": "tokenizer.json", "sha256": "tokenizer"},
        ],
        "code_sha256": {"next_iteration/grounded_graph_features.py": file_sha256(code_path)},
    }
    entry = {"id": "e0", "source_id": "7", "packet_file": "packets/e0.json",
             "packet_sha256": packet["sha256"], "status": "available"}
    arrays = {
        "source": np.arange(len(packet["nodes"]) * 4, dtype=np.float32).reshape(len(packet["nodes"]), 4),
        "query": np.arange(len(packet["target_token_ids"]) * 4, dtype=np.float32).reshape(len(packet["target_token_ids"]), 4),
    }
    np.savez(tmp_path / "features" / "e0.npz", **arrays)
    receipt = {
        "schema": "grounded-graph-feature-capture@1",
        "packet_sha256": packet["sha256"],
        "representation": REPRESENTATION,
        "model_identity": model_identity(settings),
        "source_array_shape": list(arrays["source"].shape),
        "query_array_shape": list(arrays["query"].shape),
        "source_array_sha256": hashlib.sha256(arrays["source"].tobytes()).hexdigest(),
        "query_array_sha256": hashlib.sha256(arrays["query"].tobytes()).hexdigest(),
    }
    receipt["sha256"] = digest(receipt)
    (tmp_path / "features" / "e0.json").write_text(json.dumps(receipt))
    return tokenizer, packet, settings, entry, arrays


def test_load_example_rebuilds_packet_and_validates_persisted_float32_arrays(tmp_path):
    tokenizer, packet, settings, entry, arrays = _fixture(tmp_path)

    loaded_packet, loaded_arrays, receipt = load_example(tmp_path, entry, settings, tokenizer)

    assert loaded_packet == packet
    assert receipt["packet_sha256"] == packet["sha256"]
    np.testing.assert_array_equal(loaded_arrays["source"], arrays["source"])
    np.testing.assert_array_equal(loaded_arrays["query"], arrays["query"])


def test_load_example_rejects_array_packet_entry_and_receipt_tampering(tmp_path):
    tokenizer, _, settings, entry, arrays = _fixture(tmp_path)
    changed = {**arrays, "source": arrays["source"].copy()}
    changed["source"][0, 0] += 1
    np.savez(tmp_path / "features" / "e0.npz", **changed)
    with pytest.raises(ValueError, match="bytes/shape/dtype"):
        load_example(tmp_path, entry, settings, tokenizer)

    # Restore arrays, then modify the persisted packet without updating its receipt.
    np.savez(tmp_path / "features" / "e0.npz", **arrays)
    packet_file = tmp_path / "packets" / "e0.json"
    packet = json.loads(packet_file.read_text())
    packet["input_ids"][0] += 1
    packet_file.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="exact reconstruction"):
        load_example(tmp_path, entry, settings, tokenizer)

    # A receipt can be internally rehashed but still cannot claim a different model.
    tokenizer, _, settings, entry, arrays = _fixture(tmp_path / "fresh")
    receipt_file = tmp_path / "fresh" / "features" / "e0.json"
    receipt = json.loads(receipt_file.read_text())
    receipt["model_identity"] = copy.deepcopy(receipt["model_identity"])
    receipt["model_identity"]["model_files"][0]["sha256"] = "other-model"
    receipt["sha256"] = digest({k: v for k, v in receipt.items() if k != "sha256"})
    receipt_file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt identity"):
        load_example(tmp_path / "fresh", entry, settings, tokenizer)


def test_model_identity_keeps_feature_code_hash_and_tokenizer_subset_bound(tmp_path):
    _, _, settings, _, _ = _fixture(tmp_path)
    identity = model_identity(settings)
    assert identity["capture_code_sha256"] == settings["code_sha256"]["next_iteration/grounded_graph_features.py"]
    assert [item["name"] for item in identity["tokenizer_files"]] == ["tokenizer.json"]


def test_complete_manifest_requires_every_available_entry_and_preserves_unavailable_census(tmp_path, monkeypatch):
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text("annotation-free input\n")
    inventory_path = tmp_path / "inventory"
    inventory_path.mkdir()
    (inventory_path / "manifest.json").write_text("sealed inventory manifest\n")
    entries = [
        {"id": "available", "status": "available", "tokens": 3},
        {"id": "unavailable", "status": "unavailable_length_or_source", "tokens": 5},
    ]
    settings = {
        "protocol": feature_runner.PROTOCOL,
        "code_sha256": {},
        "inventory_path": str(inventory_path),
        "inventory_input_sha256": file_sha256(input_path),
        "inventory_manifest_sha256": file_sha256(inventory_path / "manifest.json"),
        "inventory_settings_sha256": "inventory-settings",
        "source_artifacts": {},
        "parent": {"kind": "natural_response", "path": str(input_path), "input_sha256": file_sha256(input_path)},
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    (tmp_path / "entries.json").write_text(json.dumps(entries))
    settings_sha = file_sha256(tmp_path / "settings.json")
    (tmp_path / "prepare_manifest.json").write_text(json.dumps({
        "status": "prepared", "settings_sha256": settings_sha,
        "artifacts": {"entries.json": file_sha256(tmp_path / "entries.json")},
    }))
    summary = {
        "examples": 2, "counts": {"available": 1, "unavailable_length_or_source": 1},
        "tokens": 8, "actual_forwards": 1,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(feature_runner, "inventory_parent", lambda *_: ({"settings_file_sha256": "inventory-settings"}, {}))

    # A self-consistent manifest that omits the available entry is not complete.
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "prepare_manifest_sha256": file_sha256(tmp_path / "prepare_manifest.json"),
        "settings_sha256": settings_sha, "artifacts": {"summary.json": file_sha256(tmp_path / "summary.json")},
    }))
    with pytest.raises(ValueError, match="omit or add entry coordinates"):
        feature_runner.verify(tmp_path, complete=True)

    (tmp_path / "features").mkdir()
    for suffix in (".npz", ".json"):
        (tmp_path / "features" / ("available" + suffix)).write_bytes(b"captured")
    artifacts = {"summary.json": file_sha256(tmp_path / "summary.json")}
    artifacts.update({f"features/available{suffix}": file_sha256(tmp_path / "features" / ("available" + suffix))
                      for suffix in (".npz", ".json")})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "prepare_manifest_sha256": file_sha256(tmp_path / "prepare_manifest.json"),
        "settings_sha256": settings_sha, "artifacts": artifacts,
    }))
    _, loaded = feature_runner.verify(tmp_path, complete=True)
    assert loaded == entries
