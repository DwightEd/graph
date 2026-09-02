from dataclasses import fields

import torch

from experiments.evidence_route_state.graph import (
    GraphSequence,
    gram,
    mlp_relation,
    route_topology,
)


def topology(
    attention: torch.Tensor,
    values: torch.Tensor,
    *,
    query: int,
    response_start: int,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    heads = attention.shape[0]
    head_dim = values.shape[-1]
    if output_scale is None:
        output_scale = torch.ones(heads)
    metric = torch.eye(head_dim)[None].repeat(heads, 1, 1)
    metric *= output_scale[:, None, None].square()
    return route_topology(
        attention[:, None],
        values,
        metric,
        torch.tensor([query]),
        response_start,
    )[0]


def test_graph_sequence_keeps_every_structural_axis():
    tokens, layers, heads, channels, hidden = 2, 3, 2, 4, 5
    graph = GraphSequence(
        query_position=torch.tensor([6, 7]),
        prediction_position=torch.tensor([7, 8]),
        node_embedding=torch.zeros(tokens, channels, hidden),
        residual_gram=torch.zeros(tokens, layers + 1, channels, channels),
        head_write_gram=torch.zeros(tokens, layers, heads, channels, channels),
        route_topology=torch.zeros(tokens, layers, heads, channels, 7),
        mlp_relation=torch.zeros(tokens, layers, channels + 1),
        margin_contribution=torch.zeros(tokens, channels),
        valid=torch.ones(tokens, dtype=torch.bool),
    )

    assert tuple(field.name for field in fields(GraphSequence)) == (
        "query_position",
        "prediction_position",
        "node_embedding",
        "residual_gram",
        "head_write_gram",
        "route_topology",
        "mlp_relation",
        "margin_contribution",
        "valid",
    )
    assert graph.head_write_gram.shape == (2, 3, 2, 4, 4)
    assert graph.route_topology.shape == (2, 3, 2, 4, 7)


def test_gram_preserves_signed_route_direction():
    aligned = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    opposed = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])

    aligned_gram = gram(aligned)
    opposed_gram = gram(opposed)

    torch.testing.assert_close(
        aligned_gram.diagonal(dim1=-2, dim2=-1),
        opposed_gram.diagonal(dim1=-2, dim2=-1),
    )
    assert aligned_gram[0, 0, 1] == 1
    assert opposed_gram[0, 0, 1] == -1
    assert gram(aligned.to(torch.bfloat16)).dtype == torch.float32


def test_route_topology_uses_exact_gqa_capacity_and_all_causal_sources():
    attention = torch.full((4, 4), 0.25)
    values = torch.tensor(
        [
            [[[1.0], [10.0]]],
            [[[2.0], [20.0]]],
            [[[3.0], [30.0]]],
            [[[1000.0], [1000.0]]],
        ]
    )
    result = topology(
        attention,
        values,
        query=2,
        response_start=2,
        output_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    # Heads 0/1 use KV head 0; heads 2/3 use KV head 1. Source 3 is future
    # and contributes nothing despite its deliberately huge value.
    expected_capacity = torch.tensor(
        [
            [0.25, 0.50, 0.75],
            [0.50, 1.00, 1.50],
            [7.50, 15.0, 22.5],
            [10.0, 20.0, 30.0],
        ]
    )
    total = expected_capacity.sum(1)
    probability = expected_capacity / total[:, None]
    entropy = -(probability * probability.log()).sum(1)

    torch.testing.assert_close(result[:, 0, 0], total.log1p())
    torch.testing.assert_close(result[:, 0, 1], entropy.exp().log1p())
    torch.testing.assert_close(result[:, 0, 2], probability.max(1).values)
    torch.testing.assert_close(result[:, 0, 3], expected_capacity[:, :2].sum(1) / total)
    torch.testing.assert_close(result[:, 0, 4], torch.zeros(4))
    torch.testing.assert_close(result[:, 0, 5], expected_capacity[:, 2] / total)
    torch.testing.assert_close(result[:, 0, 6], torch.ones(4))

    bf16_result = route_topology(
        attention.to(torch.bfloat16)[:, None],
        values.to(torch.bfloat16),
        (torch.eye(1)[None] * torch.tensor([1.0, 4.0, 9.0, 16.0])[:, None, None]).to(
            torch.bfloat16
        ),
        torch.tensor([2]),
        response_start=2,
    )[0]
    assert bf16_result.dtype == torch.float32
    torch.testing.assert_close(bf16_result, result, rtol=2e-3, atol=2e-3)

    changed = values.clone()
    changed[1, 0, 0, 0] += 0.25
    assert not torch.equal(
        topology(
            attention,
            changed,
            query=2,
            response_start=2,
            output_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ),
        result,
    )
    future_changed = values.clone()
    future_changed[3] *= 100
    torch.testing.assert_close(
        topology(
            attention,
            future_changed,
            query=2,
            response_start=2,
            output_scale=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        ),
        result,
    )


def test_endpoint_role_changes_topology_without_changing_route_size():
    attention = torch.ones(1, 4)
    prompt_route = torch.zeros(4, 1, 1, 1)
    history_route = torch.zeros_like(prompt_route)
    prompt_route[0] = 1
    history_route[2] = 1

    prompt = topology(attention, prompt_route, query=3, response_start=2)
    history = topology(attention, history_route, query=3, response_start=2)

    torch.testing.assert_close(prompt[..., :3], history[..., :3])
    torch.testing.assert_close(prompt[..., 6], history[..., 6])
    assert prompt[0, 0, 3] == 1
    assert history[0, 0, 4] == 1
    assert not torch.equal(prompt, history)


def test_head_source_assignment_changes_head_resolved_topology():
    values = torch.ones(4, 1, 1, 1)
    first = torch.tensor([[0.8, 0.2, 0.0, 0.0], [0.1, 0.3, 0.6, 0.0]])
    second = torch.tensor([[0.7, 0.0, 0.3, 0.0], [0.2, 0.5, 0.3, 0.0]])

    first_state = topology(first, values, query=3, response_start=4)
    second_state = topology(second, values, query=3, response_start=4)

    # Per-source and per-head total capacities are identical. The graph state
    # still changes because the head/source assignment is not averaged away.
    torch.testing.assert_close(first.sum(0), second.sum(0))
    torch.testing.assert_close(first.sum(1), second.sum(1))
    assert not torch.equal(first_state, second_state)


def test_mlp_relation_keeps_signed_alignment_and_relative_size():
    registers = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    write = torch.tensor([[1.0, -1.0]])

    relation = mlp_relation(registers, write)

    root_two = torch.sqrt(torch.tensor(2.0))
    torch.testing.assert_close(relation[0, 0], 1 / root_two)
    torch.testing.assert_close(relation[0, 1], -1 / root_two)
    torch.testing.assert_close(relation[0, 2], torch.log(torch.tensor(2.0)))

    reversed_relation = mlp_relation(registers, -write)
    torch.testing.assert_close(reversed_relation[..., :2], -relation[..., :2])
    torch.testing.assert_close(reversed_relation[..., 2], relation[..., 2])


def test_zero_routes_have_zero_finite_topology_and_mlp_relation():
    result = topology(
        torch.ones(2, 3),
        torch.zeros(3, 4, 1, 2),
        query=2,
        response_start=2,
    )
    relation = mlp_relation(torch.zeros(1, 4, 3), torch.zeros(1, 3))

    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0
    assert torch.isfinite(relation).all()
    assert torch.count_nonzero(relation) == 0
