import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from next_iteration import grounded_graph_restoration_evaluate as evaluate
from next_iteration import grounded_graph_restoration_features as features
from next_iteration import grounded_graph_restoration_predict as predict


class CharTokenizer:
    bos_token_id = 101

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == " ":
            return [0]
        return [ord(ch) for ch in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        ids = [ord(ch) for ch in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


def _packet(prompt="Q? SRC END", source_span=(3, 6), response="ab"):
    tokenizer = CharTokenizer()
    prompt_ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    source_offsets = [[0, 0]]
    left, right = source_span
    for a, b in [(i, i + 1) for i in range(len(prompt))]:
        source_offsets.append([max(0, a - left), min(right - left, b - left)] if a < right and b > left else [0, 0])
    ids = prompt_ids + response_ids
    plen = len(prompt_ids)
    return {
        "sha256": "packet-sha",
        "input_ids": ids[:-1],
        "prompt_record": {"prompt": prompt, "prompt_length": plen, "source_span": list(source_span)},
        "source_offsets": source_offsets,
        "query_positions": list(range(plen - 1, len(ids) - 1)),
        "target_positions": list(range(plen, len(ids))),
        "target_token_ids": response_ids,
        "nodes": [{"type": 0, "available": True, "prompt_token_indices": [4]}],
        "edges": [],
        "response_offsets": [[0, 1], [1, 2]],
        "response_text": response,
        "weak_targets": [{"owner_node_index": 0}],
        "pointer_supervision": [{"query_index": 0, "owner_node_indices": [0]}],
    }


def test_intervention_erases_exact_positive_source_overlap_tokens_and_records_boundary_crossing():
    tokenizer = CharTokenizer()
    packet = _packet(prompt="QSRCX", source_span=(1, 4), response="ab")

    receipt = features.intervention(packet, tokenizer)

    assert receipt["erased_prompt_token_indices"] == [2, 3, 4]
    executed = receipt["executed_input_ids"]
    original = packet["input_ids"]
    assert executed[:2] == original[:2]
    assert executed[5:] == original[5:]
    assert executed[2:5] == [0, 0, 0]
    assert receipt["boundary_crossing_tokens"] == []
    assert receipt["query_positions"] == packet["query_positions"]

    bad = dict(packet)
    bad["source_offsets"] = [[0, 0] for _ in packet["source_offsets"]]
    with pytest.raises(ValueError, match="source erasure"):
        features.intervention(bad, tokenizer)


def test_intervention_receipts_whole_token_boundary_crossing_without_claiming_char_locality():
    packet = _packet(prompt="AΩB", source_span=(1, 2), response="x")
    # Simulate a tokenizer that has one prompt token spanning outside the source.
    packet["source_offsets"] = [[0, 0], [0, 0], [0, 1], [0, 0]]

    class BoundaryTokenizer(CharTokenizer):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            result = {"input_ids": [1, 2, 3]}
            if return_offsets_mapping:
                result["offset_mapping"] = [(0, 1), (0, 3), (2, 3)]
            return result

    packet["input_ids"] = [101, 1, 2, 3]
    packet["prompt_record"]["prompt_length"] = 4
    packet["query_positions"] = [3]
    packet["target_positions"] = [4]

    receipt = features.intervention(packet, BoundaryTokenizer())

    assert receipt["erased_prompt_token_indices"] == [2]
    assert receipt["boundary_crossing_tokens"] == [{"position": 2, "prompt_char_span": [0, 3]}]


class ToyAdapter:
    def __init__(self, hidden):
        self.hidden = torch.tensor(hidden, dtype=torch.float32)

    def __call__(self, **kwargs):
        query = kwargs["query_states"]
        rows = query.shape[0]
        return {
            "hidden": self.hidden[:rows].to(query.device),
            "pointer": torch.ones((rows, kwargs["source_states"].shape[0]), device=query.device),
            "gate": torch.zeros(rows, device=query.device),
            "residual": self.hidden[:rows].to(query.device) - query,
        }


def test_score_one_separates_full_gap_from_empty_restoration_differences():
    packet = {
        "nodes": [{"type": 0, "available": True}],
        "target_token_ids": [0, 1],
        "edges": [],
        "pointer_supervision": [],
    }
    arrays = {
        "source": np.zeros((1, 2), dtype=np.float32),
        "query": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        "full_query": np.array([[-3.0, 0.0], [0.0, -3.0]], dtype=np.float32),
    }
    head = torch.eye(2)
    adapters = {"graph": ToyAdapter([[0.0, 0.0], [0.0, 0.0]]), "no_edges": ToyAdapter([[3.0, 0.0], [0.0, 3.0]])}

    scores, traces = predict.score_one(adapters, head, packet, arrays, seed=11)

    assert set(scores) == set(predict.PROTOCOL["scores"])
    assert np.all(scores["graph_difference"] < 0)
    assert np.all(scores["graph_restoration_difference"] > 0)
    assert not np.allclose(scores["graph_difference"], scores["graph_restoration_difference"])
    assert np.allclose(scores["no_edges_restoration_difference"], 0.0)
    assert traces["graph_pointer"].shape == (2, 1)


def test_evaluate_ranking_uses_source_balancing_and_returns_none_for_single_class():
    y = np.array([0, 1, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    source = np.array(["a", "a", "b", "b"])

    ranked = evaluate.ranking(y, scores, source)

    assert ranked["error_prevalence"] == pytest.approx(0.75)
    assert 0.0 <= ranked["auroc"] <= 1.0
    assert evaluate.ranking(np.zeros(3, dtype=bool), np.arange(3), np.array(["a", "b", "c"])) is None


def test_evaluate_refuses_label_using_population_before_copying_evaluator(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions"
    population = tmp_path / "population"
    output = tmp_path / "evaluation"
    predictions.mkdir()
    population.mkdir()
    (predictions / "settings.json").write_text(json.dumps({"protocol": predict.PROTOCOL, "features_path": str(tmp_path / "features"), "features_manifest_sha256": "sha", "code_sha256": {}}) + "\n")
    (predictions / "manifest.json").write_text(json.dumps({"status": "complete", "settings_sha256": "pred-settings", "artifacts": {"summary.json": "sha"}}) + "\n")
    (predictions / "summary.json").write_text("{}\n")
    (population / "settings.json").write_text(json.dumps({"labels_used": True}) + "\n")
    monkeypatch.setattr(evaluate, "file_sha256", lambda path: "pred-settings" if str(path).endswith("predictions/settings.json") else "sha")
    monkeypatch.setattr(evaluate, "verify", lambda *args, **kwargs: ({}, []))

    with pytest.raises(ValueError, match="label-free measurement"):
        evaluate.run(SimpleNamespace(predictions=predictions, population=population, output=output))
    assert not output.exists()
