import torch

from experiments.holoroute import (
    Flat1024,
    HoloRoute,
    HoloRouteConfig,
    build_pairs,
    score_graph,
    self_supervised_loss,
)
from experiments.holoroute.baseline import flat_loss, score_flat
from experiments.holoroute.config import ModelConfig
from experiments.holoroute.tests.helpers import synthetic_graph


def small_config() -> HoloRouteConfig:
    return HoloRouteConfig(
        model=ModelConfig(
            hidden_dim=32,
            head_layers=1,
            head_attention_heads=4,
            head_pool_batch_size=2,
            transport_rank=4,
            message_layers=1,
        )
    )


def test_graph_model_loss_and_token_residuals():
    graph = synthetic_graph()
    config = small_config()
    assert config.train.mixed_precision is False

    model = HoloRoute(graph.layer_count, graph.head_count, config.model)
    assert model.encoder.head_pool.num_heads == config.model.head_attention_heads
    assert model.encoder.pool_batch_size == 2

    output = model(graph)
    assert output.state.dtype == torch.float32
    assert output.state.shape == (graph.event_count, 32)
    assert output.predictions.value.shape == (graph.event_count, 4, graph.head_count)
    assert output.coverage.any()
    assert output.coverage[:, 2].any()
    assert output.holonomy.shape == (1,)

    generator = torch.Generator().manual_seed(9)
    loss = self_supervised_loss(model, graph, config, generator)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    residuals = score_graph(model, graph, config, seed=17)
    assert residuals.value.shape == (graph.response_count, 6)
    assert residuals.coverage.shape == residuals.value.shape


def test_flat_baseline_uses_same_layers_without_graph_edges():
    graph = synthetic_graph()
    config = small_config()
    pairs = build_pairs(graph)
    model = Flat1024(graph.layer_count, graph.head_count, hidden=32, blocks=1)

    generator = torch.Generator().manual_seed(11)
    loss = flat_loss(model, pairs, config, generator)
    assert torch.isfinite(loss)
    loss.backward()

    residuals = score_flat(model, pairs, config, seed=19)
    assert residuals.value.shape == (graph.response_count, 1)
    assert residuals.coverage.shape == residuals.value.shape
