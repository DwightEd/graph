import numpy as np
import torch

from experiments.non_neural_structure_audit.nulls import (
    EndpointSwapPlan,
    constrained_endpoint_swap,
)

from .helpers import routing_state


def _lag_bin(query, source, response_idx):
    lag = query - (source - response_idx)
    return torch.floor(torch.log2(lag.float())).long()


def test_constrained_swap_preserves_role_lag_degree_and_target_weights():
    routing = routing_state(
        layers=[0] * 4,
        heads=[0] * 4,
        queries=[6, 7, 8, 9],
        sources=[1, 3, 3, 5],
        weights=[0.11, 0.12, 0.13, 0.14],
        diagonal=torch.zeros((10, 1, 1)),
        response_tokens=10,
        num_layers=1,
    )
    edges = routing.edges

    result = constrained_endpoint_swap(edges, seed=7, rounds=20)
    rewired = result.edges

    assert result.changed_fraction >= 0.75
    torch.testing.assert_close(rewired.weight, edges.weight)
    torch.testing.assert_close(
        _lag_bin(rewired.query, rewired.source, edges.response_idx),
        _lag_bin(edges.query, edges.source, edges.response_idx),
    )
    assert torch.equal(
        torch.sort(rewired.source).values,
        torch.sort(edges.source).values,
    )
    assert bool((rewired.source < edges.response_idx + rewired.query).all())
    assert result.audit["duplicate_edges"] == 0
    assert result.audit["causal_violations"] == 0


def test_lineage_endpoint_null_does_not_count_or_rewire_prompt_edges():
    routing = routing_state(
        layers=[0, 0, 0, 0],
        heads=[0, 0, 0, 0],
        queries=[6, 7, 8, 9],
        sources=[0, 3, 3, 5],
        weights=[0.11, 0.12, 0.13, 0.14],
        diagonal=torch.zeros((10, 1, 1)),
        response_tokens=10,
        num_layers=1,
    )

    result = constrained_endpoint_swap(routing.edges, seed=3, rounds=20)

    assert result.edges.source[0] == 0
    assert result.audit["eligible_response_edges"] == 3


def test_endpoint_plan_uses_compact_integer_geometry():
    routing = routing_state(
        layers=[0, 0],
        heads=[0, 0],
        queries=[6, 7],
        sources=[1, 3],
        weights=[0.1, 0.2],
        diagonal=torch.zeros((8, 1, 1)),
        response_tokens=8,
        num_layers=1,
    )

    plan = EndpointSwapPlan(routing.edges)

    assert plan.rows.dtype == np.int32
    assert plan.original.dtype == np.int32
    assert plan.response_edges.dtype == np.int32
    assert sum(group.stop - group.start for group in plan.group_slices) == len(
        plan.response_edges
    )
