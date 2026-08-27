from dataclasses import replace
import math

import torch

from experiments.directed_route_hypergraph.routing_dispersion import (
    DIAGONAL_MASS,
    LOWER,
    PROMPT_MASS,
    RESPONSE_MASS,
    UNRESOLVED_MASS,
    UPPER,
    attention_routing_dispersion,
)
from experiments.grounded_route.graph import TokenEdges, TokenGraph


def routing_graph(
    *,
    response_start: int,
    response_count: int,
    heads: int,
    source: list[int],
    target: list[int],
    head: list[int],
    weight: list[float],
    diagonal: torch.Tensor,
    unresolved: torch.Tensor,
) -> TokenGraph:
    count = len(source)
    return TokenGraph(
        sample_id="dispersion",
        source_id="source",
        task_type="QA",
        response_start=response_start,
        token_count=response_start + response_count,
        response_count=response_count,
        layer_count=1,
        head_count=heads,
        attention_floor=0.5,
        edges=TokenEdges(
            source=torch.tensor(source, dtype=torch.long),
            target=torch.tensor(target, dtype=torch.long),
            layer=torch.zeros(count, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            weight=torch.tensor(weight),
        ),
        diagonal=diagonal,
        unresolved=unresolved,
        token_ids=torch.arange(100, 100 + response_start + response_count),
    ).check().canonicalize()


def test_exact_concentrated_endpoint_has_zero_entropy_and_unit_hhi():
    graph = routing_graph(
        response_start=2,
        response_count=1,
        heads=1,
        source=[0],
        target=[2],
        head=[0],
        weight=[1.0],
        diagonal=torch.zeros((1, 1, 1)),
        unresolved=torch.zeros((1, 1, 1)),
    )

    output = attention_routing_dispersion(graph)

    assert torch.equal(
        output.per_head.normalized_entropy[0, 0, 0], torch.tensor([0.0, 0.0])
    )
    assert torch.equal(output.per_head.hhi[0, 0, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(
        output.per_head.normalized_hhi[0, 0, 0], torch.tensor([1.0, 1.0])
    )
    assert output.per_head.censored_endpoint_count[0, 0, 0] == 1


def test_exact_uniform_endpoint_distribution_has_unit_normalized_entropy():
    third = 1.0 / 3.0
    graph = routing_graph(
        response_start=2,
        response_count=1,
        heads=1,
        source=[0, 1],
        target=[2, 2],
        head=[0, 0],
        weight=[third, third],
        diagonal=torch.full((1, 1, 1), third),
        unresolved=torch.zeros((1, 1, 1)),
    )

    output = attention_routing_dispersion(graph)

    assert torch.allclose(
        output.per_head.normalized_entropy[0, 0, 0],
        torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(
        output.per_head.hhi[0, 0, 0],
        torch.full((2,), third),
        atol=1e-6,
    )
    assert torch.allclose(
        output.per_head.normalized_hhi[0, 0, 0],
        torch.zeros(2),
        atol=1e-6,
    )
    assert output.per_head.censored_endpoint_count[0, 0, 0] == 0


def test_unresolved_mass_produces_concentration_and_uniform_bounds():
    graph = routing_graph(
        response_start=3,
        response_count=1,
        heads=1,
        source=[0],
        target=[3],
        head=[0],
        weight=[0.5],
        diagonal=torch.full((1, 1, 1), 0.1),
        unresolved=torch.full((1, 1, 1), 0.4),
    )

    output = attention_routing_dispersion(graph)
    bounds = output.per_head
    known_entropy = -0.5 * math.log(0.5) - 0.1 * math.log(0.1)
    lower = (known_entropy - 0.4 * math.log(0.4)) / math.log(4.0)
    upper = (known_entropy - 2.0 * 0.2 * math.log(0.2)) / math.log(4.0)

    assert bounds.censored_endpoint_count[0, 0, 0] == 2
    assert torch.allclose(
        bounds.normalized_entropy[0, 0, 0],
        torch.tensor([lower, upper]),
        atol=1e-6,
    )
    assert torch.allclose(
        bounds.hhi[0, 0, 0],
        torch.tensor([0.34, 0.42]),
        atol=1e-6,
    )
    assert torch.allclose(
        bounds.normalized_hhi[0, 0, 0],
        torch.tensor([(0.34 - 0.25) / 0.75, (0.42 - 0.25) / 0.75]),
        atol=1e-6,
    )


def test_role_mass_and_censoring_preserve_head_specific_routing():
    graph = routing_graph(
        response_start=2,
        response_count=2,
        heads=2,
        source=[0, 2, 1, 2],
        target=[3, 3, 3, 3],
        head=[0, 0, 1, 1],
        weight=[0.4, 0.3, 0.1, 0.2],
        diagonal=torch.tensor([[[1.0, 1.0]], [[0.2, 0.3]]]),
        unresolved=torch.tensor([[[0.0, 0.0]], [[0.1, 0.4]]]),
    )

    output = attention_routing_dispersion(graph)

    assert torch.allclose(
        output.head_role_mass[1, 0],
        torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]),
    )
    assert torch.allclose(
        output.query_role_mass[1, 0], torch.tensor([0.25, 0.25, 0.25, 0.25])
    )
    assert torch.allclose(
        output.query_role_mass_disagreement[1, 0],
        torch.tensor([0.15, 0.05, 0.05, 0.15]),
    )
    assert output.query_role_js_disagreement[1, 0] > 0
    assert output.query_role_js_disagreement[1, 0] <= 1
    assert torch.equal(
        output.per_head.censored_endpoint_count[1, 0], torch.tensor([1, 1])
    )
    assert torch.allclose(
        output.query_entropy_bounds,
        output.per_head.normalized_entropy.mean(dim=2),
    )


def test_predecessor_query_alignment_omits_both_unobserved_boundaries():
    graph = routing_graph(
        response_start=2,
        response_count=3,
        heads=1,
        source=[],
        target=[],
        head=[],
        weight=[],
        diagonal=torch.ones((3, 1, 1)),
        unresolved=torch.zeros((3, 1, 1)),
    )
    graph = replace(graph, token_ids=torch.tensor([10, 11, 20, 21, 22]))

    output = attention_routing_dispersion(graph)

    assert torch.equal(output.predictor_response_index, torch.tensor([0, 1]))
    assert torch.equal(output.token_response_index, torch.tensor([1, 2]))
    assert torch.equal(output.token_id, torch.tensor([21, 22]))
    assert torch.equal(output.token_entropy_bounds, output.query_entropy_bounds[:-1])
    assert torch.equal(
        output.token_concentration_bounds,
        output.query_concentration_bounds[:-1],
    )
    assert output.token_entropy_bounds.shape == (2, 1, 2)
    assert output.head_role_mass[0, 0, 0, DIAGONAL_MASS] == 1.0
    assert output.head_role_mass[0, 0, 0, PROMPT_MASS] == 0.0
    assert output.head_role_mass[0, 0, 0, RESPONSE_MASS] == 0.0
    assert output.head_role_mass[0, 0, 0, UNRESOLVED_MASS] == 0.0
