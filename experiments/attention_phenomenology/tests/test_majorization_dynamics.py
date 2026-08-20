import pytest
import torch

from experiments.attention_phenomenology.majorization_dynamics import (
    CausalStateFilter,
    causal_route_trace,
    causal_route_trace_from_edges,
    exact_prompt_routes,
)
from experiments.attention_phenomenology.routing import RoutingEdges


def test_route_trace_is_prefix_invariant_and_detects_concentration():
    probability = torch.tensor(
        [
            [[[0.50, 0.30, 0.20]]],
            [[[0.50, 0.30, 0.20]]],
            [[[0.90, 0.05, 0.05]]],
            [[[0.90, 0.05, 0.05]]],
        ]
    )
    full = causal_route_trace(probability, history_decay=0.8)
    prefix = causal_route_trace(probability[:3], history_decay=0.8)

    torch.testing.assert_close(full.majorization_evidence[:3], prefix.majorization_evidence)
    torch.testing.assert_close(full.concentration_level[:3], prefix.concentration_level)
    assert full.majorization_evidence[2].item() > 0
    assert full.concentration_level[2].item() > 0
    assert full.source_affinity[3].item() > full.source_affinity[2].item()


def test_causal_filter_separates_entry_from_stable_concentrated_residence():
    observations = torch.tensor(
        [
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [2.0, 2.0, 0.1],
            [2.0, 2.0, 0.95],
            [2.0, 2.0, 0.98],
        ]
    )
    state_filter = CausalStateFilter()

    states = state_filter.run(observations)
    prefix = state_filter.run(observations[:4])

    assert states.entry_probability[2] > states.basin_probability[2]
    assert states.basin_probability[4] > states.entry_probability[4]
    torch.testing.assert_close(states.forecast_probability[:4], prefix.forecast_probability)


def test_causal_filter_rejects_negative_transition_probability():
    transition = torch.tensor(
        [[1.1, -0.1, 0.0], [0.1, 0.2, 0.7], [0.1, 0.1, 0.8]]
    )

    with pytest.raises(ValueError, match="non-negative"):
        CausalStateFilter(transition)


def test_exact_prompt_routes_use_floor_excess_and_keep_heads_separate():
    edges = RoutingEdges(
        num_layers=1,
        num_heads=2,
        num_response_tokens=1,
        num_tokens=4,
        response_idx=3,
        attention_floor=0.1,
        layer=torch.tensor([0, 0, 0]),
        head=torch.tensor([0, 0, 1]),
        query=torch.tensor([0, 0, 0]),
        source=torch.tensor([0, 1, 2]),
        weight=torch.tensor([0.6, 0.3, 0.4]),
        diagonal=torch.zeros((1, 1, 2)),
    )

    routes = exact_prompt_routes(edges, token=0)

    torch.testing.assert_close(
        routes.probability[0, 0],
        torch.tensor([5.0 / 7.0, 2.0 / 7.0, 0.0]),
    )
    torch.testing.assert_close(
        routes.probability[0, 1],
        torch.tensor([0.0, 0.0, 1.0]),
    )
    torch.testing.assert_close(routes.excess_mass[0], torch.tensor([0.7, 0.3]))


def test_streamed_edge_trace_matches_readable_probability_tensor_path():
    probability = torch.tensor(
        [
            [[[0.5, 0.3, 0.2]]],
            [[[0.9, 0.05, 0.05]]],
            [[[0.9, 0.05, 0.05]]],
        ]
    )
    query, source = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
    edges = RoutingEdges(
        num_layers=1,
        num_heads=1,
        num_response_tokens=3,
        num_tokens=6,
        response_idx=3,
        attention_floor=0.1,
        layer=torch.zeros(9, dtype=torch.long),
        head=torch.zeros(9, dtype=torch.long),
        query=query.reshape(-1),
        source=source.reshape(-1),
        weight=probability[:, 0, 0].reshape(-1) + 0.1,
        diagonal=torch.zeros((3, 1, 1)),
    )

    dense = causal_route_trace(probability, history_decay=0.8)
    streamed = causal_route_trace_from_edges(edges, history_decay=0.8)

    torch.testing.assert_close(streamed.majorization_evidence, dense.majorization_evidence)
    torch.testing.assert_close(streamed.concentration_level, dense.concentration_level)
    torch.testing.assert_close(streamed.source_affinity, dense.source_affinity)
