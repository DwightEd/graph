import torch

from experiments.graph_structure_audit.graph_data import (
    build_multiplex_graph,
    load_multiplex_graph,
)
from .helpers import Sample, raw_graph


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


def test_loading_graph_releases_sample_attention_before_returning():
    raw, labels = raw_graph()
    sample = Sample(raw, labels)

    graph = load_multiplex_graph(sample, block_rows=8)

    assert graph.num_edges > 0
    assert sample._attention is None
    assert sample.release_calls == 1
