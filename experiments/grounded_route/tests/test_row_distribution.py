import torch

from experiments.grounded_route.config import LearningConfig, ModelConfig
from experiments.grounded_route.learning import (
    edge_rows,
    sample_route_rows,
    segment_log_softmax,
    self_supervised_loss,
)
from experiments.grounded_route.model import GroundedRouteEncoder
from experiments.grounded_route.tests.helpers import make_graph


def test_route_row_sampler_keeps_complete_rows():
    graph = make_graph().canonicalize()
    selected = sample_route_rows(
        graph,
        limit=2,
        generator=torch.Generator().manual_seed(7),
    )
    rows = edge_rows(graph)
    for local_row in range(selected.count):
        edge = selected.edge[selected.row == local_row]
        row = rows[edge[0]]
        expected = torch.nonzero(rows == row, as_tuple=False).flatten()
        assert torch.equal(edge, expected)


def test_segment_log_softmax_normalizes_each_attention_row():
    score = torch.tensor([0.0, 1.0, -1.0, 2.0])
    group = torch.tensor([0, 0, 1, 1])
    log_probability = segment_log_softmax(score, group, group_count=2)
    probability = log_probability.exp()

    assert torch.allclose(probability[group == 0].sum(), torch.tensor(1.0))
    assert torch.allclose(probability[group == 1].sum(), torch.tensor(1.0))


def test_row_distribution_objective_backpropagates():
    torch.manual_seed(11)
    graph = make_graph()
    model = GroundedRouteEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(
            hidden_dim=24,
            edge_hidden_dim=32,
            lag_buckets=8,
            dropout=0.0,
        ),
    ).train()
    output = self_supervised_loss(
        model,
        graph,
        LearningConfig(
            objective="row_distribution",
            route_rows_per_graph=4,
            negative_count=2,
            negative_attempt_factor=8,
            variance_weight=0.05,
        ),
        torch.Generator().manual_seed(13),
    )

    assert output.row_count > 0
    assert output.pair_count > 0
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.route_query[1].weight.grad is not None
    assert model.edge_message[1].weight.grad is not None
