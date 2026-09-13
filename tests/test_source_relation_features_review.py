"""Review tests for SourceRel feature inventory boundaries."""

import json
from pathlib import Path

import pytest

from next_iteration.source_relation_features import PROTOCOL, prepare_inventory, verify_prepared
from next_iteration.source_relation_train import source_items
from route_graph.frozen_reader import digest


def _model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir(exist_ok=True)
    (path / "config.json").write_text("{}\n")
    (path / "tokenizer.json").write_text("{}\n")
    return path


def _doc(text, ids, status="encodable"):
    return {"id": digest(text), "text": text, "input_ids": ids, "token_count": len(ids), "status": status}


def test_prepare_inventory_retains_over_limit_documents_without_truncation(tmp_path):
    short = _doc("short view", [1, 2, 3])
    long_ids = list(range(PROTOCOL["max_tokens"] + 1))
    long = _doc("long view", long_ids, "unavailable_length")
    bindings = [{"source_id": "1", "split": "source_train", "queries": [], "candidates": [
        {"id": "short", "text": short["text"], "document_id": short["id"]},
        {"id": "long", "text": long["text"], "document_id": long["id"]},
    ]}]

    settings = prepare_inventory(tmp_path / "features", _model_dir(tmp_path), [short, long], bindings, {"kind": "fixture"})
    _, docs, loaded_bindings = verify_prepared(tmp_path / "features")

    assert settings["length_preflight"]["documents"] == 2
    assert settings["length_preflight"]["unavailable_documents"] == 1
    saved_long = next(d for d in docs if d["id"] == long["id"])
    assert saved_long["input_ids"] == long_ids
    assert saved_long["status"] == "unavailable_length"
    assert loaded_bindings == bindings


def test_prepare_inventory_is_fresh_only_and_binds_document_digest(tmp_path):
    doc = _doc("short view", [1, 2])
    output = tmp_path / "features"
    prepare_inventory(output, _model_dir(tmp_path), [doc], [{"source_id": "1", "split": "source_train", "queries": [], "candidates": []}], {"kind": "fixture"})
    with pytest.raises(ValueError, match="fresh output"):
        prepare_inventory(output, _model_dir(tmp_path), [doc], [], {"kind": "fixture"})

    documents_path = output / "documents.json"
    tampered = json.loads(documents_path.read_text())
    tampered[0]["input_ids"] = [99]
    documents_path.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="prepared feature inventory changed"):
        verify_prepared(output)


def test_unavailable_documents_exclude_or_mark_the_whole_query_pool(tmp_path):
    query = _doc("query view", [1, 2])
    positive = _doc("positive view", list(range(PROTOCOL["max_tokens"] + 1)), "unavailable_length")
    negative = _doc("negative view", [3, 4])
    bindings = [{"source_id": "1", "split": "source_train", "queries": [{
        "id": "q1", "text": query["text"], "document_id": query["id"],
        "positive_ids": ["p1"], "candidate_ids": ["p1", "n1"],
    }], "candidates": [
        {"id": "p1", "text": positive["text"], "document_id": positive["id"]},
        {"id": "n1", "text": negative["text"], "document_id": negative["id"]},
    ]}]

    prepare_inventory(tmp_path / "features", _model_dir(tmp_path), [query, positive, negative], bindings, {"kind": "fixture"})
    _, docs, loaded_bindings = verify_prepared(tmp_path / "features")

    unavailable_docs = {doc["id"] for doc in docs if doc["status"] != "encodable"}
    candidates = {c["id"]: c for binding in loaded_bindings for c in binding["candidates"]}
    affected = []
    for binding in loaded_bindings:
        for q in binding["queries"]:
            referenced = {q["document_id"], *(candidates[cid]["document_id"] for cid in q["candidate_ids"])}
            if referenced & unavailable_docs:
                affected.append(q)
    assert not affected or all(q.get("feature_status") == "unavailable_length" for q in affected)


def test_unavailable_feature_status_is_written_and_trainer_excludes_whole_pool(tmp_path):
    query = _doc("query view", [1, 2])
    positive = _doc("positive view", list(range(PROTOCOL["max_tokens"] + 1)), "unavailable_length")
    negative = _doc("negative view", [3, 4])
    bindings = [{"source_id": "1", "split": "source_train", "queries": [{
        "id": "q1", "text": query["text"], "document_id": query["id"],
        "value_type": "text", "positive_ids": ["p1"], "candidate_ids": ["p1", "n1"],
        "homograph_negative_ids": [], "other_record_same_field_ids": [],
    }], "candidates": [
        {"id": "p1", "text": positive["text"], "document_id": positive["id"], "value_type": "text"},
        {"id": "n1", "text": negative["text"], "document_id": negative["id"], "value_type": "text"},
    ]}]

    prepare_inventory(tmp_path / "features", _model_dir(tmp_path), [query, positive, negative], bindings, {"kind": "fixture"})
    _, docs, loaded_bindings = verify_prepared(tmp_path / "features")

    assert loaded_bindings[0]["queries"][0]["feature_status"] == "unavailable_length"
    items, excluded = source_items(docs, loaded_bindings)
    assert items == []
    assert excluded == [{"source_id": "1", "split": "source_train",
        "query_id": "q1", "status": "whole_query_pool_unavailable_length"}]
