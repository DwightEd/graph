import torch

from experiments.evidence_route_state.graph import (
    ResponseGraphBuilder,
    attention_write,
    route_totals,
    sparsify_route_chunk,
    sparsify_routes,
)

from .helpers import route_row


def test_sparse_unknown_tail_conserves_write_capacity_and_support():
    messages = torch.tensor([[[1.0, 0.0], [0.4, 0.0], [0.3, 0.0]]])
    capacity = messages.norm(dim=-1)
    support = torch.tensor([[0.5, 0.2, 0.1]])
    head_write = messages.sum(dim=1)

    row = sparsify_routes(
        capacity,
        support,
        head_write,
        lambda head, source: messages[head, source],
        layer=0,
        query_position=4,
        coverage=0.5,
        max_edges_per_head=1,
    )

    assert row.source.tolist() == [0]
    torch.testing.assert_close(attention_write(row), head_write.sum(dim=0))
    total_capacity, total_support, total_positive = route_totals(row)
    torch.testing.assert_close(total_capacity, capacity.sum(dim=1))
    torch.testing.assert_close(total_support, support.sum(dim=1))
    torch.testing.assert_close(total_positive, support.clamp_min(0).sum(dim=1))

    # Omitted mass has no invented source endpoint; it survives only in the
    # endpoint-free unknown account.
    torch.testing.assert_close(row.unknown_capacity, torch.tensor([0.7]))
    torch.testing.assert_close(row.unknown_positive_support, torch.tensor([0.3]))


def test_vector_edges_preserve_cross_head_cancellation():
    messages = torch.tensor([[[2.0, 0.0]], [[-2.0, 0.0]]])
    capacity = messages.norm(dim=-1)
    support = torch.tensor([[2.0], [-2.0]])
    head_write = messages.sum(dim=1)

    row = sparsify_routes(
        capacity,
        support,
        head_write,
        lambda head, source: messages[head, source],
        layer=0,
        query_position=2,
        coverage=1.0,
        max_edges_per_head=1,
    )

    torch.testing.assert_close(attention_write(row), torch.zeros(2))
    assert row.capacity.sum() == 4.0
    assert sorted(row.head.tolist()) == [0, 1]


def test_chunk_sparsification_conserves_every_query_row():
    messages = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.4, 0.1], [0.2, -0.1]],
                [[0.1, 0.6], [0.0, 0.2], [-0.3, 0.1]],
            ],
            [
                [[0.7, 0.2], [0.3, 0.0], [0.1, 0.1]],
                [[0.0, 0.8], [0.2, 0.3], [-0.1, 0.2]],
            ],
        ]
    )
    capacity = messages.norm(dim=-1)
    state = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    support = torch.einsum("qhsd,qd->qhs", messages, state)
    support /= state.square().sum(-1)[:, None, None]
    head_write = messages.sum(dim=2)

    rows = sparsify_route_chunk(
        capacity,
        support,
        head_write,
        lambda query, head, source: messages[query, head, source],
        layer=3,
        query_position=torch.tensor([7, 8]),
        coverage=0.55,
        max_edges_per_head=2,
    )

    assert [row.query_position for row in rows] == [7, 8]
    for query, row in enumerate(rows):
        torch.testing.assert_close(attention_write(row), head_write[query].sum(0))
        total_capacity, total_support, total_positive = route_totals(row)
        torch.testing.assert_close(total_capacity, capacity[query].sum(-1))
        torch.testing.assert_close(total_support, support[query].sum(-1))
        torch.testing.assert_close(
            total_positive,
            support[query].clamp_min(0).sum(-1),
        )


def test_sparse_budget_is_applied_to_each_head():
    messages = torch.tensor(
        [
            [[2.0, 0.0], [1.0, 0.0]],
            [[0.0, 3.0], [0.0, 1.0]],
        ]
    )
    capacity = messages.norm(dim=-1)
    support = capacity / capacity.sum()

    row = sparsify_routes(
        capacity,
        support,
        messages.sum(1),
        lambda head, source: messages[head, source],
        layer=0,
        query_position=2,
        coverage=0.5,
        max_edges_per_head=1,
    )

    assert row.head.tolist() == [0, 1]
    assert row.source.tolist() == [0, 0]


def test_response_graph_persists_exact_rows_endpoints_and_offsets():
    builder = ResponseGraphBuilder(response_start=3)
    rows = [
        route_row(
            0,
            1,
            source=(0,),
            support=(1.0,),
            residual_support=0.0,
        ),
        route_row(
            0,
            2,
            source=(0, 2),
            support=(0.6, 0.4),
            residual_support=0.0,
        ),
        route_row(
            1,
            3,
            source=(3,),
            support=(1.0,),
            residual_support=0.0,
        ),
        route_row(1, 4),
    ]
    builder.add_many(rows)

    graph = builder.finish()

    assert graph.row_layer.tolist() == [0, 1, 1]
    assert graph.row_query_position.tolist() == [2, 3, 4]
    assert graph.row_prediction_position.tolist() == [3, 4, 5]
    assert graph.edge_start.tolist() == [0, 2, 3, 3]
    assert graph.edge_source.tolist() == [0, 2, 3]
    assert graph.edge_head.tolist() == [0, 0, 0]
    assert graph.row_layer.dtype == torch.int16
    assert graph.row_query_position.dtype == torch.int32
    assert graph.edge_source.dtype == torch.int32
    assert graph.edge_head.dtype == torch.int16
    torch.testing.assert_close(graph.edge_capacity, torch.tensor([0.6, 0.4, 1.0]))
    torch.testing.assert_close(graph.edge_support, torch.tensor([0.6, 0.4, 1.0]))
    torch.testing.assert_close(
        graph.reconstructed_head_write_norm[:, 0],
        torch.tensor([1.0, 1.0, 0.0]),
    )
    assert graph.reconstructed_head_write_norm.shape == (3, 1)
    assert graph.reconstructed_attention_write_norm.shape == (3,)
    torch.testing.assert_close(
        graph.reconstructed_attention_write_norm,
        torch.tensor([1.0, 1.0, 0.0]),
    )
