import torch

from experiments.attention_mechanism_audit.graph import (
    GraphBuilder,
    PROFILE_CHANNELS,
    ROLE_NAMES,
    TOKEN_FLOW_CHANNELS,
    source_roles,
)


def test_factorized_edge_function_matches_materialized_avwo():
    torch.manual_seed(3)
    heads, kv_heads, head_dim, hidden, sources = 4, 2, 3, 12, 7
    attention = torch.softmax(torch.randn(heads, sources), -1)
    value = torch.randn(sources, kv_heads, head_dim)
    q_to_kv = torch.tensor([0, 0, 1, 1])
    output_weight = torch.randn(hidden, heads * head_dim)
    gradient = torch.randn(hidden)

    value_by_head = value[:, q_to_kv]
    head_message = attention.T[..., None] * value_by_head
    blocks = output_weight.reshape(hidden, heads, head_dim).permute(1, 2, 0)
    residual_message = torch.einsum("shd,hdo->hso", head_message, blocks)
    expected_function = torch.einsum("hso,o->hs", residual_message, gradient)
    expected_norm = residual_message.norm(dim=-1)

    head_gradient = torch.nn.functional.linear(
        gradient, output_weight.T
    ).reshape(heads, head_dim)
    actual_function = attention * torch.einsum(
        "shd,hd->hs", value_by_head, head_gradient
    )
    block_gram = torch.einsum("hdo,heo->hde", blocks, blocks)
    actual_norm = torch.einsum(
        "shd,hde,she->hs", head_message, block_gram, head_message
    ).clamp_min(0).sqrt()

    torch.testing.assert_close(actual_function, expected_function)
    torch.testing.assert_close(actual_norm, expected_norm)


def test_dense_profile_uses_all_edges_and_selected_edges_close_with_tail():
    torch.manual_seed(5)
    token_ids = torch.arange(8)
    builder = GraphBuilder(
        token_ids,
        response_start=5,
        layers=1,
        heads=2,
        head_dim=3,
        edge_cover=0.5,
        edge_budget=2,
    )
    attention = torch.softmax(torch.randn(2, 5), -1)
    value = torch.randn(5, 1, 3)
    q_to_kv = torch.tensor([0, 0])
    output_weight = torch.randn(6, 6)
    block = output_weight.reshape(6, 2, 3)
    block_gram = torch.einsum("ohd,ohe->hde", block, block)
    gradient = torch.randn(6)
    roles = source_roles(5, 5, 4, torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool))
    builder.add_layer(
        target=0,
        predictor=4,
        layer=0,
        attention=attention,
        value=value,
        head_gradient=torch.nn.functional.linear(
            gradient, output_weight.T
        ).reshape(2, 3),
        output_gram=block_gram,
        q_to_kv=q_to_kv,
        roles=roles,
        mlp_output=torch.randn(6),
        mlp_gradient=torch.randn(6),
    )
    graph = builder.finish()

    assert graph.node_profile.shape == (
        3,
        1,
        2,
        len(ROLE_NAMES),
        len(PROFILE_CHANNELS),
    )
    assert graph.edge_index.shape[1] <= 2
    assert graph.edge_head_message.shape == (graph.edge_index.shape[1], 3)
    assert graph.token_flow.shape == (3, 8, len(TOKEN_FLOW_CHANNELS))

    value_by_head = value[:, q_to_kv]
    head_gradient = torch.nn.functional.linear(
        gradient, output_weight.T
    ).reshape(2, 3)
    function = attention * torch.einsum(
        "shd,hd->hs", value_by_head, head_gradient
    )
    head_message = attention.T[..., None] * value_by_head
    transport = torch.einsum(
        "shd,hde,she->hs", head_message, block_gram, head_message
    ).clamp_min(0).sqrt()
    expected_flow = torch.stack(
        (
            function.clamp_min(0).sum(dim=0),
            (-function).clamp_min(0).sum(dim=0),
            attention.sum(dim=0),
            transport.sum(dim=0),
        ),
        dim=-1,
    )
    torch.testing.assert_close(graph.token_flow[0, :5], expected_flow)
    assert torch.count_nonzero(graph.token_flow[0, 5:]) == 0

    selected = torch.zeros_like(graph.node_profile[0, 0].float())
    for edge in range(graph.edge_index.shape[1]):
        head = int(graph.edge_head[edge])
        role = int(graph.edge_role[edge])
        function = graph.edge_function[edge]
        selected[head, role, 0] += graph.edge_attention[edge]
        selected[head, role, 1] += graph.edge_residual_norm[edge]
        selected[head, role, 2] += function.clamp_min(0)
        selected[head, role, 3] += (-function).clamp_min(0)

        block = output_weight.reshape(6, 2, 3)[:, head]
        residual = block @ graph.edge_head_message[edge].float()
        torch.testing.assert_close(
            residual.norm(), graph.edge_residual_norm[edge].float(), atol=2e-3, rtol=0
        )
        torch.testing.assert_close(
            residual @ gradient,
            graph.edge_function[edge],
            atol=2e-4,
            rtol=2e-4,
        )

    torch.testing.assert_close(
        selected + graph.edge_tail_profile[0, 0].float(),
        graph.node_profile[0, 0].float(),
        atol=2e-3,
        rtol=0,
    )


def test_source_roles_separate_evidence_history_and_predictor_self():
    evidence = torch.tensor([False, True, True, False, False])
    roles = source_roles(8, response_start=5, predictor=7, evidence_mask=evidence)
    assert roles.tolist() == [1, 0, 0, 1, 1, 2, 2, 3]
