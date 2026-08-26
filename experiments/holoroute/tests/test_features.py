import torch

from experiments.holoroute.config import FeatureConfig
from experiments.holoroute.features import build_node_features
from experiments.holoroute.tests.helpers import synthetic_graph


def test_graph_structure_is_embedded_in_node_features():
    graph = synthetic_graph()
    config = FeatureConfig(source_basis_dim=4, head_projection_dim=2)
    features = build_node_features(graph, config)

    per_layer = min(graph.head_count, config.head_projection_dim) * (
        3 * config.source_basis_dim + 2
    )
    assert features.token_layer.shape == (
        graph.response_count,
        graph.layer_count,
        per_layer,
    )
    assert features.node.shape == (
        graph.response_count,
        graph.layer_count * per_layer,
    )
    assert features.inherited_prompt.abs().sum() > 0
    assert torch.isfinite(features.node).all()
