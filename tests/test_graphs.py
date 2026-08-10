import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import attention_entries
from graphs import (
    build_hypergraph,
    build_multiplex,
    build_original,
    build_relation_topk,
    build_support,
)


def sample():
    return {
        "sample_id": "1",
        "source_id": "s",
        "response_idx": 3,
        "token_ids": torch.arange(5),
        "attention_diagonal": torch.ones((1, 2, 5)) * 0.1,
        "row_ptr": torch.tensor([0, 2, 4, 6, 8]),
        "source_index": torch.tensor([0, 2, 0, 3, 1, 2, 2, 3]),
        "attention_weight": torch.tensor([0.6, 0.2, 0.4, 0.3, 0.7, 0.1, 0.5, 0.2]),
        "attention_floor": 0.01,
    }


def test_entries_decode_layer_head_target():
    e = attention_entries(sample())
    assert e["layer"].tolist() == [0] * 8
    assert e["head"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert e["target"].tolist() == [3, 3, 4, 4, 3, 3, 4, 4]


def test_original_threshold_union_semantics():
    g = build_original(sample(), tau=0.25)
    pairs = list(map(tuple, g["edge_index"].t().tolist()))
    assert pairs == [(0, 3), (1, 3), (0, 4), (2, 4), (3, 4)]
    assert g["edge_attr"].shape == (5, 2)
    assert g["edge_type"].tolist() == [0, 0, 0, 0, 1]


def test_other_graph_views_run():
    assert build_multiplex(sample(), tau=0.25)["edge_index"].shape[1] == 5
    assert build_support(sample(), mass=0.5)["edge_index"].shape[1] > 0
    assert build_relation_topk(sample(), 1, 1)["edge_index"].shape[1] > 0
    h = build_hypergraph(sample(), tau=0.25)
    assert h["hyperedge_target"].numel() > 0
    assert h["incidence_index"].shape[0] == 2
