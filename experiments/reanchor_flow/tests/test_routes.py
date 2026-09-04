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
        distance_scale=2,
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
    assert result.nonlocality.shape == (2, 3)
    assert result.route_change.shape == (2, 3)
    assert result.predictor_reuse.shape == (2, 3)
    assert result.future_influence.shape == (2, 3)
    assert result.head["route_change"].shape == (2, 2, 3)
    assert result.head["attention_evidence_mass"].shape == (2, 2, 3)
    assert result.head["route_change"].dtype == torch.float16
    assert result.detail["edge_map"].shape == (3, 5)
    assert result.detail["nonlocal_head"].shape == (2, 2, 3)
    assert torch.isfinite(result.future_influence[:, :2]).all()


def test_predictor_reuse_is_not_emitted_token_anchor():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
    accumulator = RouteAccumulator(
        model,
        response_start=3,
        prompt_evidence_mask=[True, False, False],
        route_window=1,
        future_horizon=2,
        distance_scale=2,
    )
    probability = torch.zeros(1, 1, 6, 6)
    for query in range(6):
        probability[0, 0, query, query] = 1
    # Event 0 is predicted at q=2 and emits p=3.  Both future rows reuse q=2
    # while never reading p=3, so the two coordinates must disagree.
    probability[0, 0, 3:5] = 0
    probability[0, 0, 3:5, 2] = 1
    value = torch.ones(1, 1, 6, 2)
    accumulator.observe(0, probability, value, torch.eye(2))
    result = accumulator.finish()
    torch.testing.assert_close(result.predictor_reuse[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(result.future_influence[0, 0], torch.tensor(0.0))


def test_nonlocality_is_continuous_expected_distance():
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
    accumulator = RouteAccumulator(
        model,
        response_start=2,
        prompt_evidence_mask=[True, False],
        route_window=1,
        future_horizon=2,
        distance_scale=2,
    )
    probability = torch.zeros(1, 1, 4, 4)
    probability[0, 0, 0, 0] = 1
    probability[0, 0, 1, 1] = 1
    probability[0, 0, 2, 2] = 1
    probability[0, 0, 3, 1] = 0.5
    probability[0, 0, 3, 2] = 0.5
    value = torch.ones(1, 1, 4, 2)
    output = torch.eye(2)
    accumulator.observe(0, probability, value, output)
    result = accumulator.finish()
    # At q=3, lag 1 receives weight 0.5 and lag 2 receives weight 1.0.
    torch.testing.assert_close(result.nonlocality[0, 2], torch.tensor(0.75))
