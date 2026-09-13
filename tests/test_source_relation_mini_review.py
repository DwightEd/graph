"""Review tests for SourceRel-Mini data/model boundaries."""

import argparse
import hashlib
import json

import pytest
import torch

from next_iteration.source_relation_data import compile_source, prepare
from next_iteration.source_relation_model import SourceRelMini, owner_beam, owner_nce, source_role_mass


def _row(source, *, response="response", response_sha=None):
    prompt = "Task. Source: " + source
    if response_sha is None:
        response_sha = hashlib.sha256(response.encode()).hexdigest()
    return {
        "id": "123",
        "source_id": "456",
        "task": "Data2txt",
        "generator": "fixture",
        "official_split": "train",
        "prompt": prompt,
        "source_span": [prompt.index(source), prompt.index(source) + len(source)],
        "response": response,
        "response_sha256": response_sha,
        "prompt_length": 1,
        "token_ids": [1],
        "source_mask": [False],
        "offsets": [],
    }


def test_compile_source_uses_actual_root_and_keeps_homograph_as_negative():
    source = repr({
        "name": "Cafe",
        "attributes": {"WiFi": "free", "Parking": "free", "Mood": "paid"},
    })

    compiled = compile_source(source, "456")

    assert compiled["record_root_id"] == compiled["source_graph"]["literal_graph"]["root"]
    root_node = next(n for n in compiled["source_graph"]["literal_graph"]["nodes"] if n["id"] == compiled["record_root_id"])
    assert root_node["path"] == []
    wifi_query = next(q for q in compiled["queries"] if q["family"] == "attributes / Wi Fi")
    assert '"WiFi": "<VALUE>"' in wifi_query["text"]
    assert '"Parking": "free"' in wifi_query["text"]
    assert wifi_query["homograph_negative_ids"]


def test_duplicate_masked_contexts_are_multipositive_without_list_indices_in_text():
    source = repr({
        "name": "Cafe",
        "business_stars": 3.0,
        "review_info": [
            {"review_stars": 4.0, "review_date": "same", "review_text": "ok"},
            {"review_stars": 4.0, "review_date": "same", "review_text": "ok"},
        ],
    })

    compiled = compile_source(source, "456")

    review_query = next(q for q in compiled["queries"] if q["family"] == "review info / review stars")
    assert len(review_query["positive_ids"]) == 2
    assert "review info / 0" not in review_query["text"]
    assert "review info / 1" not in review_query["text"]
    candidates = {c["id"]: c for c in compiled["candidates"]}
    for cid in review_query["candidate_ids"]:
        assert "review info / 0" not in candidates[cid]["text"]
        assert "review info / 1" not in candidates[cid]["text"]


def test_prepare_rejects_frozen_row_response_hash_mismatch(tmp_path):
    source = repr({"name": "Cafe", "attributes": {"WiFi": "free", "Parking": "paid"}})
    bad_row = _row(source, response="actual response", response_sha="0" * 64)
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps(bad_row) + "\n")

    with pytest.raises(ValueError, match="response text/hash mismatch"):
        prepare(argparse.Namespace(inputs=inputs, output=tmp_path / "out"))


def test_model_masks_invalid_candidates_and_outputs_reusable_candidate_beam():
    model = SourceRelMini(input_dim=4, hidden_dim=2, num_types=2, temperature=1.0)
    with torch.no_grad():
        model.query.weight.zero_()
        model.candidate.weight.zero_()
        model.query.weight[0, 0] = 1.0
        model.candidate.weight[0, 0] = 1.0
        model.candidate.weight[1, 2] = 1.0
    queries = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    candidates = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[True, False, True]])
    scores = model(queries, candidates, torch.tensor([0]), valid=valid)
    assert scores[0, 0] > scores[0, 2]
    assert torch.isneginf(scores[0, 1])
    positives = torch.tensor([[True, False, False]])
    loss = owner_nce(scores, positives)
    assert torch.isfinite(loss)
    beam = owner_beam(scores, topk=3)[0]
    assert beam["candidate_indices"] == [0, 2]
    assert beam["source_reuse_allowed"] is True
    families, mass = source_role_mass(scores, ["a", "b", "a"])
    assert families == ["a", "b"]
    assert torch.allclose(mass.sum(-1), torch.ones(1))
