from dataclasses import dataclass

import numpy as np
import torch

from experiments.information_flow import encode


@dataclass(frozen=True)
class Graph:
    sample_id: str
    source_id: str
    task_type: str
    response_start: int
    layer_count: int
    head_count: int
    token_ids: torch.Tensor
    node_embedding: torch.Tensor
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_head: torch.Tensor
    edge_weight: torch.Tensor
    diagonal: torch.Tensor
    unresolved: torch.Tensor

    @property
    def response_count(self):
        return len(self.token_ids) - self.response_start

    @property
    def response_embedding(self):
        return self.node_embedding[self.response_start :]


class Record:
    def __init__(self, graph):
        self.graph = graph

    def load(self):
        return self.graph


def test_encode_bundle_saves_row_aligned_node_data(tmp_path, monkeypatch):
    graph = Graph(
        sample_id="sample",
        source_id="source",
        task_type="QA",
        response_start=1,
        layer_count=1,
        head_count=1,
        token_ids=torch.tensor([10, 11]),
        node_embedding=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_layer=torch.tensor([0]),
        edge_head=torch.tensor([0]),
        edge_weight=torch.tensor([0.75]),
        diagonal=torch.tensor([[[0.25]]]),
        unresolved=torch.zeros(1, 1, 1),
    )
    bundle = type("Bundle", (), {"records": (Record(graph),)})()
    monkeypatch.setattr(encode, "load_bundle", lambda _: bundle)

    report = encode.encode_bundle(
        "source.npz",
        tmp_path,
        mode="mean",
        checkpoints=1,
    )
    with np.load(tmp_path / "index.npz", allow_pickle=False) as data:
        assert data["embedding"].shape == (1, 4)
        assert data["sample_id"].tolist() == ["sample"]
        assert not bool(data["labels_included"].item())
    assert report["nodes"] == 1
