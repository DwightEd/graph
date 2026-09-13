"""CPU contract checks for source-field reconstruction training."""

import torch

from next_iteration.source_relation_model import SourceRelMini, owner_nce
from next_iteration.source_relation_train import (
    batch_masks,
    lexical_scores,
    source_items,
)


def _doc(identifier, text, status="encodable"):
    return {"id": identifier, "text": text, "status": status}


def _bindings():
    return [
        {
            "source_id": "train-source",
            "split": "source_train",
            "candidates": [
                {"id": "ta", "document_id": "ta", "value_type": "text"},
                {"id": "tb", "document_id": "tb", "value_type": "text"},
            ],
            "queries": [
                {"id": "train-query", "document_id": "tq", "positive_ids": ["ta"],
                 "candidate_ids": ["ta", "tb"], "value_type": "text",
                 "homograph_negative_ids": [], "other_record_same_field_ids": []},
                # This full pool is unavailable and must affect neither training
                # examples nor the lexical vocabulary.
                {"id": "unavailable-query", "document_id": "over-limit", "positive_ids": ["ta"],
                 "candidate_ids": ["ta", "tb"], "value_type": "text",
                 "homograph_negative_ids": [], "other_record_same_field_ids": []},
            ],
        },
        {
            "source_id": "validation-source",
            "split": "source_validation",
            "candidates": [
                {"id": "va", "document_id": "va", "value_type": "text"},
                {"id": "vb", "document_id": "vb", "value_type": "text"},
            ],
            "queries": [
                {"id": "validation-query", "document_id": "vq", "positive_ids": ["va"],
                 "candidate_ids": ["va", "vb"], "value_type": "text",
                 "homograph_negative_ids": [], "other_record_same_field_ids": []},
            ],
        },
    ]


def _docs():
    return [
        _doc("tq", "train query words"),
        _doc("ta", "train candidate alpha"),
        _doc("tb", "train candidate beta"),
        _doc("over-limit", "unavailable vocabulary leak token", "unavailable_length"),
        _doc("vq", "validation only vocabulary"),
        _doc("va", "validation candidate alpha"),
        _doc("vb", "validation candidate beta"),
    ]


def test_tfidf_fit_uses_only_available_train_query_pools_and_not_validation_views():
    docs, bindings = _docs(), _bindings()
    items, excluded = source_items(docs, bindings)

    records, receipt = lexical_scores(docs, bindings, items)

    assert {item["source_id"] for item in items} == {"train-source", "validation-source"}
    assert excluded == [{"source_id": "train-source", "split": "source_train", "query_id": "unavailable-query",
                         "status": "whole_query_pool_unavailable_length"}]
    # q + two candidates is the complete available training pool.  The unavailable
    # query and every validation view must be absent from the fitted vocabulary.
    assert receipt["fit_document_count"] == 3
    assert not receipt["fit_uses_source_validation"]
    assert {record["split"] for record in records} == {"source_train", "source_validation"}


def test_source_batch_masks_block_cross_source_candidates_and_keep_gradients_finite():
    items, _ = source_items(_docs(), _bindings())
    qidx, cidx, types, valid, positive = batch_masks(items)

    assert valid.shape == positive.shape == (2, 4)
    assert torch.equal(valid, torch.tensor([[True, True, False, False], [False, False, True, True]]))
    assert torch.equal(positive, torch.tensor([[True, False, False, False], [False, False, True, False]]))

    torch.manual_seed(3)
    model = SourceRelMini(input_dim=5, hidden_dim=3, num_types=7, temperature=.07)
    vectors = torch.randn(7, 5)
    scores = model(vectors[qidx], vectors[cidx], types, valid=valid)
    loss = owner_nce(scores, positive)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
