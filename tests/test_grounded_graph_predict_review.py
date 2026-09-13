import json

import numpy as np
import pytest
import torch

from next_iteration import grounded_graph_predict as predict
from next_iteration.grounded_graph_features import EDGE_KINDS, NODE_KINDS


def test_words_keep_all_nonspace_units_and_weight_token_scores_by_char_overlap():
    packet = {
        "response_text": "Hi Café!",
        "response_offsets": [[0, 2], [3, 7], [7, 8]],
    }
    scores = {
        "base_nll": np.array([1.0, 2.0, 10.0], dtype=np.float32),
        "base_entropy": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    available = predict.words(packet, scores, "available")
    assert [w["text"] for w in available] == ["Hi", "Café!"]
    assert available[0]["scores"]["base_nll"] == pytest.approx(1.0)
    assert available[1]["scores"]["base_nll"] == pytest.approx((2.0 * 4 + 10.0 * 1) / 5)
    assert all(w["status"] == "available" for w in available)

    unavailable = predict.words(packet, scores, "unavailable_length_or_source")
    assert [w["text"] for w in unavailable] == ["Hi", "Café!"]
    assert all(w["status"] == "unavailable" for w in unavailable)
    assert all(all(value == 0.0 for value in w["scores"].values()) for w in unavailable)


class ToyAdapter:
    def __init__(self, hidden_transform):
        self.hidden_transform = hidden_transform

    def __call__(self, **kwargs):
        query = kwargs["query_states"]
        hidden = self.hidden_transform(query)
        return {
            "hidden": hidden,
            "pointer": torch.ones((query.shape[0], kwargs["source_states"].shape[0]), device=query.device),
            "gate": torch.zeros(query.shape[0], device=query.device),
            "residual": hidden - query,
        }


def test_score_one_uses_logp_base_minus_adapter_direction_and_full_score_set():
    packet = {
        "nodes": [{"type": 0, "available": True}],
        "target_token_ids": [0, 1],
        "edges": [],
        "pointer_supervision": [],
    }
    arrays = {
        "source": np.zeros((1, 2), dtype=np.float32),
        "query": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    }
    head = torch.eye(2)
    adapters = {
        "graph": ToyAdapter(lambda query: -query),
        "no_edges": ToyAdapter(lambda query: query),
    }

    scores, traces = predict.score_one(adapters, head, packet, arrays, seed=7)

    assert set(scores) == set(predict.PROTOCOL["scores"])
    assert np.all(scores["base_nll"] >= 0)
    assert np.all(scores["graph_difference"] > 0)
    assert np.allclose(scores["no_edges_difference"], 0.0)
    assert traces["graph_pointer"].shape == (2, 1)
    assert traces["no_edges_residual_norm"].shape == (2,)


def test_permuted_arguments_only_changes_edge_destinations_without_mutating_original():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    arguments = {
        "edge_index": edge_index,
        "node_types": torch.tensor([0, 1, 2, 3]),
    }

    permuted = predict.permuted_arguments(arguments, seed=123)

    assert arguments["edge_index"].tolist() == [[0, 1, 2], [1, 2, 3]]
    assert permuted["edge_index"] is not edge_index
    assert permuted["edge_index"][0].tolist() == edge_index[0].tolist()
    generator = torch.Generator(device=edge_index.device).manual_seed(123)
    expected_destination = torch.randperm(4, generator=generator)[edge_index[1]]
    assert permuted["edge_index"][1].tolist() == expected_destination.tolist()
    assert all(0 <= value < len(arguments["node_types"]) for value in permuted["edge_index"][1].tolist())
    assert permuted["node_types"] is arguments["node_types"]


def test_load_adapter_rejects_checkpoint_vocabulary_or_training_settings_mismatch(tmp_path, monkeypatch):
    training = tmp_path / "training"
    training.mkdir()
    checkpoint = training / "graph" / "initial.pt"
    checkpoint.parent.mkdir()
    module = torch.nn.Linear(1, 1)
    torch.save({
        "state_dict": module.state_dict(),
        "epoch": 0,
        "arm": "graph",
        "settings_sha256": "settings-sha",
        "node_kinds": ["wrong"],
        "edge_kinds": EDGE_KINDS,
    }, checkpoint)
    (training / "settings.json").write_text("settings\n")
    monkeypatch.setattr(predict, "file_sha256", lambda path: "settings-sha" if str(path).endswith("settings.json") else "checkpoint-sha")
    monkeypatch.setattr(predict, "trained", lambda path: ({}, {
        "arms": {"graph": {"selected": {"epoch": 0, "checkpoint": "graph/initial.pt", "checkpoint_sha256": "checkpoint-sha"}}}
    }, {}))
    monkeypatch.setattr(predict, "new_adapter", lambda device: torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="vocabulary|settings mismatch"):
        predict.load_adapter(training, "graph", "cpu")

    torch.save({
        "state_dict": module.state_dict(),
        "epoch": 0,
        "arm": "graph",
        "settings_sha256": "settings-sha",
        "node_kinds": NODE_KINDS,
        "edge_kinds": EDGE_KINDS,
    }, checkpoint)
    loaded = predict.load_adapter(training, "graph", "cpu")
    assert isinstance(loaded, torch.nn.Linear)
    assert loaded.training is False
    assert not any(p.requires_grad for p in loaded.parameters())


def test_run_rejects_non_natural_feature_parent_before_gpu_or_label_access(tmp_path, monkeypatch):
    output = tmp_path / "prediction"
    training = tmp_path / "training"
    features = tmp_path / "features"
    training.mkdir()
    features.mkdir()
    monkeypatch.setattr(predict, "trained", lambda path: ({"code_sha256": {}, "source_text_sha256": {"train": [], "validation": []}}, {"arms": {}}, {"model_files": []}))
    monkeypatch.setattr(predict, "verify", lambda *args, **kwargs: ({"parent": {"kind": "source_reconstruction"}, "model_files": []}, []))

    with pytest.raises(ValueError, match="natural inference observer"):
        predict.run(type("Args", (), {"output": output, "training": training, "features": features})())
    assert not output.exists()
