import torch

from experiments.holoroute.config import HoloRouteConfig, ModelConfig
from experiments.holoroute.model import HoloRouteEncoder
from experiments.holoroute.objectives import score_graph, self_supervised_loss
from experiments.holoroute.tests.helpers import synthetic_graph


def test_model_uses_event_path_depth_and_query_structure():
    graph = synthetic_graph()
    config = HoloRouteConfig(model=ModelConfig(hidden_dim=32, head_encoder_heads=4, transport_rank=4))
    model = HoloRouteEncoder(graph.num_layers, graph.num_heads, config.model)
    output = model(graph)
    assert output.state.shape == (graph.num_events, 32)
    assert output.event_prediction.shape == graph.event_head_value.shape
    assert output.depth_coverage.any()
    assert output.relay_coverage.any()
    assert output.query_coverage.any()
    assert output.holonomy_error.shape == (1,)

    no_relay = model(graph, relay_keep=torch.zeros(graph.relay_edge_index.shape[1], dtype=torch.bool))
    assert not torch.allclose(output.state, no_relay.state)


def test_self_supervised_loss_backpropagates_and_scores_locally():
    graph = synthetic_graph()
    config = HoloRouteConfig(model=ModelConfig(hidden_dim=32, head_encoder_heads=4, transport_rank=4))
    model = HoloRouteEncoder(graph.num_layers, graph.num_heads, config.model)
    generator = torch.Generator().manual_seed(9)
    loss = self_supervised_loss(model, graph, config, generator=generator)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    feature, coverage = score_graph(model, graph, config, seed=17)
    assert feature.shape == (graph.num_response_tokens, 6)
    assert coverage.shape == feature.shape
    assert coverage[:, 0].sum() > 0
