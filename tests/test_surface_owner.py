from pathlib import Path

import numpy as np
import pytest

import next_iteration.surface_owner as module
from next_iteration.surface_graph import source_occurrences, surface_graph
from next_iteration.surface_owner import (
    OWNER_PROTOCOL,
    _array_sha,
    capture_masked_contexts,
    document,
    feature_requests,
    match_owners,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest

IDENTITY = {"test_fixture": "not_model_results"}


def receipt_fixture(docs, vectors):
    """Only a unit fixture for the consumer's integrity checks, never a run."""
    result = {"schema": "masked-owner-capture@1", "model_identity": IDENTITY,
        "layers": OWNER_PROTOCOL["layers"], "representation": OWNER_PROTOCOL["features"],
        "code_sha256": file_sha256(Path(module.__file__)), "hidden_size": 8,
        "records": [{"document": d, "input_ids": [1], "input_ids_sha256": digest([1]),
            "model_identity": IDENTITY, "layers": OWNER_PROTOCOL["layers"],
            "representation": OWNER_PROTOCOL["features"], "array_sha256": _array_sha(vectors[d["document_id"]])} for d in docs]}
    return {**result, "sha256": digest(result)}


def fixtures(value="5.0"):
    response = surface_graph("The cafe is rated " + value + ".", sample_id="r", side="response")
    source = source_occurrences("[{'name': 'A', 'stars': 4.0}, {'name': 'B', 'stars': 4.0}]", sample_id="s", task="Data2txt")
    return response, source


def numeric(match, graph):
    sid = next(s["slot_id"] for s in graph["slots"] if s["surface_type"] == "number")
    return next(m for m in match["matches"] if m["slot_id"] == sid)


def test_no_target_value_or_target_hidden_state_in_owner_ranking():
    a, source = fixtures("1.0")
    b, _ = fixtures("99.999")
    first = numeric(match_owners(a, source, {}, mode="lexical_only_preflight"), a)
    second = numeric(match_owners(b, source, {}, mode="lexical_only_preflight"), b)
    assert first["selected"] == second["selected"]
    assert first["target_document_id"] == second["target_document_id"]
    assert first["owner_ambiguous"] and first["candidate_count"] == 2


def test_dense_capture_required_and_lexical_mode_explicit():
    response, source = fixtures()
    with pytest.raises(ValueError, match="capture records"):
        match_owners(response, source, {})
    result = match_owners(response, source, {}, mode="lexical_only_preflight")
    assert result["dense_feature_stage_executed"] is False
    assert result["native_forward_calls"] == 0


def test_dense_vectors_change_proposal_without_truth_claim():
    response, source = fixtures()
    docs = feature_requests(response, source)["documents"]
    vectors = {d["document_id"]: np.ones((3, 8), dtype=np.float32) for d in docs}
    numbers = [o for o in source["occurrences"] if o["surface_type"] == "number"]
    vectors[document(numbers[1]["masked_context"])["document_id"]] *= -1
    capture = receipt_fixture(docs, vectors)
    match = numeric(match_owners(response, source, vectors, capture=capture, expected_model_identity=IDENTITY), response)
    assert match["selected"][0]["occurrence_id"] == numbers[0]["occurrence_id"]
    assert not match["owner_ambiguous"] and match["support_status"] == "not_assessed"


def test_all_competing_occurrences_remain_in_denominator_beyond_topk():
    response, _ = fixtures()
    source = source_occurrences(repr([{"stars": 4.0}] * 7), sample_id="s", task="Data2txt")
    match = numeric(match_owners(response, source, {}, mode="lexical_only_preflight"), response)
    assert match["candidate_count"] == 7 and len(match["selected"]) == 4
    assert match["unsearched_count"] == 3 and len(match["near_tie_occurrence_ids"]) == 7
    assert all(c["same_source_value_occurrences"] == 7 for c in match["selected"])


@pytest.mark.parametrize("bad", [np.zeros((3, 8)), np.full((3, 8), np.nan), np.ones((2, 8))])
def test_invalid_dense_features_rejected(bad):
    response, source = fixtures()
    docs = feature_requests(response, source)["documents"]
    vectors = {d["document_id"]: bad for d in docs}
    capture = receipt_fixture(docs, vectors)
    with pytest.raises(ValueError):
        match_owners(response, source, vectors, capture=capture, expected_model_identity=IDENTITY)


def test_dense_array_and_model_identity_tampering_rejected():
    response, source = fixtures()
    docs = feature_requests(response, source)["documents"]
    vectors = {d["document_id"]: np.ones((3, 8), dtype=np.float32) for d in docs}
    capture = receipt_fixture(docs, vectors)
    with pytest.raises(ValueError, match="identity"):
        match_owners(response, source, vectors, capture=capture, expected_model_identity={"different": "model"})
    vectors[docs[0]["document_id"]][0, 0] += 1
    with pytest.raises(ValueError, match="array/document"):
        match_owners(response, source, vectors, capture=capture, expected_model_identity=IDENTITY)


def test_tiny_random_cpu_model_batch_padding_and_hook_cleanup():
    """Actual CPU forwards of a random tiny model, not natural research results."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text, add_special_tokens):
            return [1] + [3 + ord(c) % 29 for c in text]

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                                        num_hidden_layers=24, num_attention_heads=2, num_key_value_heads=2)).eval()
    docs = [document("The rating is <NUMBER>."), document("The much longer restaurant name and its rating is <NUMBER>.")]
    batched, capture = capture_masked_contexts(model, Tokenizer(), docs, model_identity=IDENTITY, batch_size=2)
    singles, _ = capture_masked_contexts(model, Tokenizer(), docs, model_identity=IDENTITY, batch_size=1)
    assert capture["feature_forward_calls"] == 1 and capture["native_forward_calls"] == 0
    for did in batched:
        np.testing.assert_allclose(batched[did], singles[did], atol=1e-6, rtol=1e-5)
    assert all(not layer._forward_hooks for layer in model.model.layers)
    with pytest.raises(ValueError, match="no silent truncation"):
        capture_masked_contexts(model, Tokenizer(), docs, model_identity=IDENTITY, max_tokens=3)
    assert all(not layer._forward_hooks for layer in model.model.layers)
