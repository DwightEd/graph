from types import SimpleNamespace

import torch

from experiments.reanchor_flow.message_norm import source_norm
from experiments.reanchor_flow.routes import RouteAccumulator


class DummyModel:
    config = SimpleNamespace(num_hidden_layers=2)


def test_source_norm_matches_direct_projection():
    torch.manual_seed(7)
    heads, sources, head_dim, hidden = 4, 6, 3, 12
    value = torch.randn(heads, sources, head_dim)
    output = torch.randn(hidden, heads * head_dim)
    actual = source_norm(value, output)
    blocks = output.reshape(hidden, heads, head_dim).permute(1, 2, 0)
    expected = torch.stack(
        [
            torch.linalg.vector_norm(value[h] @ blocks[h], dim=-1)
            for h in range(heads)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_accumulator_keeps_token_and_layer_trajectories():
    accumulator = RouteAccumulator(
        DummyModel(),
        response_start=3,
        prompt_evidence_mask=[True, False, False],
        route_window=2,
        future_horizon=2,
        far_lag=1,
        detail=True,
    )
    probability = torch.zeros(1, 2, 5, 5)
    for query in range(5):
        probability[:, :, query, : query + 1] = 1.0 / (query + 1)
    value = torch.ones(1, 2, 5, 2)
    output = torch.eye(4)
    for layer in range(2):
        accumulator.observe(layer, probability, value, output)
    result = accumulator.finish()
    assert result.prompt_share.shape == (2, 3)
    assert result.route_change.shape == (2, 3)
    assert result.future_influence.shape == (2, 3)
    assert result.detail["edge_map"].shape == (3, 5)
    assert torch.isfinite(result.future_influence[:, :2]).all()
