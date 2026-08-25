from __future__ import annotations

import torch

from experiments.holoroute.flat1024 import (
    Flat1024Config,
    Flat1024Model,
    apply_layer_block_mask,
    build_flat_pair_view,
    flat1024_loss,
    sample_layer_block_mask,
    score_flat1024,
)
from experiments.holoroute.tests.helpers import synthetic_graph


def test_flat1024_keeps_all_layers_without_graph_adjacency() -> None:
    graph = synthetic_graph()
    view = build_flat_pair_view(graph)
    assert view.num_pairs == 5
    assert view.value.shape == (5, graph.num_layers, graph.num_heads)
    assert int(view.layer_present.sum().item()) == graph.num_events
    assert view.flat_dim == graph.num_layers * graph.num_heads

    model = Flat1024Model(view.num_layers, view.num_heads)
    generator = torch.Generator().manual_seed(11)
    block_mask = sample_layer_block_mask(
        view,
        fraction=0.3,
        minimum=1,
        generator=generator,
    )
    value, observed = apply_layer_block_mask(view, block_mask)
    prediction = model(view, value=value, observed=observed)
    assert prediction.shape == view.value.shape


def test_flat1024_training_and_scoring_cover_every_existing_block() -> None:
    graph = synthetic_graph()
    view = build_flat_pair_view(graph)
    model = Flat1024Model(view.num_layers, view.num_heads)
    loss = flat1024_loss(
        model,
        view,
        Flat1024Config(),
        generator=torch.Generator().manual_seed(13),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    feature, coverage = score_flat1024(model, view, rounds=3, seed=17)
    assert feature.shape == (graph.num_response_tokens, 1)
    assert coverage.shape == feature.shape
    assert int(coverage.sum()) == int(view.layer_present.sum().item())
