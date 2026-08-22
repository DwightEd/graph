import torch

from experiments.graph_structure_audit.graph_data import build_multiplex_graph
from .helpers import raw_graph


def test_graph_preserves_exact_layer_head_tensor():
    raw, _ = raw_graph()
    graph = build_multiplex_graph(raw)
    assert graph.edge_attr.ndim == 3
    assert graph.edge_attr.shape[1:] == (raw.num_layers, raw.num_heads)
    assert graph.edge_observed.sum() == raw.num_edges

    target = raw.response_idx + raw.query[0]
    pair = torch.nonzero(
        (graph.edge_index[0] == raw.source[0]) & (graph.edge_index[1] == target),
        as_tuple=False,
    ).item()
    assert graph.edge_attr[pair, raw.layer[0], raw.head[0]] == raw.weight[0]
