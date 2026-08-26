from dataclasses import replace

import torch
from torch import nn

from experiments.dbgnn_reference.config import DBGNNConfig
from experiments.dbgnn_reference.learning import self_supervised_loss
from experiments.dbgnn_reference.tests.test_graph import encoded_graph


class FakeLinkPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        self.seen_graph = None
        self.scored_endpoints: list[tuple[torch.Tensor, torch.Tensor]] = []

    def encode(self, graph):
        self.seen_graph = graph
        return self.projection(graph.x_fo)

    def edge_score(self, embedding, source, target):
        self.scored_endpoints.append((source.detach().cpu(), target.detach().cpu()))
        return (embedding[source] * embedding[target]).sum(dim=-1)


def training_graph():
    graph = encoded_graph()
    return replace(
        graph,
        edge_index=torch.cat((graph.edge_index, torch.tensor([[0], [5]])), dim=1),
        edge_layer=torch.cat((graph.edge_layer, torch.tensor([0]))),
        edge_head=torch.cat((graph.edge_head, torch.tensor([0]))),
        edge_weight=torch.cat((graph.edge_weight, torch.tensor([0.1]))),
    )


def test_holdout_removes_every_typed_copy_and_higher_order_path(monkeypatch):
    import experiments.dbgnn_reference.learning as learning

    built_from = []
    original_build = learning.build_dbgnn_graph

    def capture(graph, **kwargs):
        built_from.append(graph)
        return original_build(graph, **kwargs)

    monkeypatch.setattr(learning, "build_dbgnn_graph", capture)
    model = FakeLinkPredictor()
    config = DBGNNConfig(
        edge_drop_fraction=1.0,
        positives_per_graph=32,
        variance_weight=0.1,
    )
    output = self_supervised_loss(
        model,
        training_graph(),
        config,
        torch.Generator().manual_seed(7),
    )

    assert output.positive_count == 1
    assert output.eligible_count == 1
    positive_source, positive_target = model.scored_endpoints[0]
    negative_source, negative_target = model.scored_endpoints[1]
    assert torch.equal(positive_source, torch.tensor([3]))
    assert torch.equal(positive_target, torch.tensor([5]))
    assert torch.equal(negative_source, torch.tensor([2]))
    assert torch.equal(negative_target, positive_target)
    assert (positive_source < 2).eq(negative_source < 2).all()
    assert torch.equal(
        torch.floor(torch.log2((positive_target - positive_source).float())),
        torch.floor(torch.log2((negative_target - negative_source).float())),
    )

    masked_source, masked_target = built_from[-1].edge_index
    assert not bool(((masked_source == 3) & (masked_target == 5)).any())
    assert len(masked_source) == len(training_graph().edge_weight) - 2

    higher_order_pairs = model.seen_graph.ho_endpoints.T
    assert not bool((higher_order_pairs == torch.tensor([3, 5])).all(dim=1).any())
    if model.seen_graph.edge_index.numel():
        path_nodes = model.seen_graph.edge_index.flatten()
        assert not bool(
            (higher_order_pairs[path_nodes] == torch.tensor([3, 5]))
            .all(dim=1)
            .any()
        )


def test_loss_is_weighted_bpr_plus_response_variance():
    torch.manual_seed(3)
    model = FakeLinkPredictor()
    config = DBGNNConfig(
        edge_drop_fraction=1.0,
        positives_per_graph=32,
        variance_weight=0.2,
    )
    output = self_supervised_loss(
        model,
        training_graph(),
        config,
        torch.Generator().manual_seed(11),
    )

    source, target = model.scored_endpoints[0]
    negative_source, _ = model.scored_endpoints[1]
    embedding = model.projection(model.seen_graph.x_fo)
    positive = (embedding[source] * embedding[target]).sum(dim=-1)
    negative = (embedding[negative_source] * embedding[target]).sum(dim=-1)
    expected_route = torch.nn.functional.softplus(negative - positive).mean()

    assert torch.allclose(output.route_loss, expected_route)
    assert torch.allclose(
        output.loss,
        output.route_loss + config.variance_weight * output.variance_loss,
    )
    output.loss.backward()
    assert model.projection.weight.grad is not None
    assert bool(torch.isfinite(model.projection.weight.grad).all())
