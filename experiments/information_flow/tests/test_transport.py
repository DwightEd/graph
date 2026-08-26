from dataclasses import dataclass

import torch

from experiments.information_flow.config import FlowConfig
from experiments.information_flow.transport import attention_output, encode_views


@dataclass(frozen=True)
class Edges:
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    weight: torch.Tensor

    @property
    def count(self):
        return int(self.source.numel())

    def select(self, mask):
        return Edges(
            self.source[mask],
            self.target[mask],
            self.layer[mask],
            self.head[mask],
            self.weight[mask],
        )

    def to(self, device):
        return Edges(
            self.source.to(device),
            self.target.to(device),
            self.layer.to(device),
            self.head.to(device),
            self.weight.to(device),
        )


@dataclass(frozen=True)
class TinyGraph:
    response_start: int
    token_count: int
    response_count: int
    layer_count: int
    head_count: int
    edges: Edges
    diagonal: torch.Tensor
    unresolved: torch.Tensor

    @property
    def device(self):
        return self.diagonal.device

    def layer_edges(self, layer, device=None):
        edges = self.edges.select(self.edges.layer == layer)
        return edges if device is None else edges.to(device)


def make_graph():
    edges = Edges(
        source=torch.tensor([0, 0, 1, 0, 1]),
        target=torch.tensor([1, 2, 2, 1, 2]),
        layer=torch.tensor([0, 0, 0, 1, 1]),
        head=torch.zeros(5, dtype=torch.long),
        weight=torch.tensor([0.5, 0.2, 0.3, 0.1, 0.7]),
    )
    diagonal = torch.tensor(
        [
            [[0.3], [0.6]],
            [[0.2], [0.1]],
        ]
    )
    unresolved = torch.tensor(
        [
            [[0.2], [0.3]],
            [[0.3], [0.2]],
        ]
    )
    return TinyGraph(1, 3, 2, 2, 1, edges, diagonal, unresolved)


def test_attention_output_matches_dense_transport():
    graph = make_graph()
    state = torch.eye(3)
    output = attention_output(graph, state, 0, "self")

    expected = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.2, 0.3, 0.5],
        ]
    )
    assert torch.allclose(output, expected, atol=1e-6)


def test_ordered_flow_exports_matched_node_views():
    graph = make_graph()
    views = encode_views(
        graph,
        FlowConfig(sketch_dim=8, residual_weight=1.0, unresolved="self"),
    )

    assert views.full_trace.shape == (2, 16)
    assert views.full_final.shape == (2, 8)
    assert views.reverse_trace.shape == (2, 16)
    assert views.trajectory.shape == (2, 2, 8)
    assert not torch.allclose(views.full_final, views.reverse_final)
    assert not torch.allclose(views.full_trace[:, :8], views.full_trace[:, 8:])
