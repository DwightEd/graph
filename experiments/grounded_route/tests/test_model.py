from dataclasses import replace

import torch

from experiments.grounded_route.config import LearningConfig, ModelConfig
from experiments.grounded_route.controls import rewire_endpoints_keep_roles
from experiments.grounded_route.graph import TokenEdges
from experiments.grounded_route.learning import matched_negative_edges, self_supervised_loss
from experiments.grounded_route.model import GroundedRouteEncoder, lag_bucket
from experiments.grounded_route.tests.helpers import (
    make_graph,
    make_rewirable_graph,
    permute_edge_storage,
)


def make_model(graph):
    model = GroundedRouteEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(
            hidden_dim=24,
            edge_hidden_dim=32,
            lag_buckets=8,
            dropout=0.0,
            head_transition_identity_bias=2.0,
        ),
    )
    return model.eval()


def change_current_and_future_rows(graph, current: int):
    selected = graph.edges.target >= current
    source = graph.edges.source.clone()
    prompt = selected & (source < graph.response_start)
    response = selected & (source >= graph.response_start)
    source[prompt] = (source[prompt] + 1).remainder(graph.response_start)
    source[response] = graph.edges.target[response] - 1
    return replace(
        graph,
        edges=TokenEdges(
            source=source,
            target=graph.edges.target,
            layer=graph.edges.layer,
            head=graph.edges.head,
            weight=graph.edges.weight,
        ),
    ).check()


def test_encoder_returns_exactly_one_embedding_per_token():
    graph = make_graph()
    output = make_model(graph).encode(graph)

    assert output.node_embedding.shape == (graph.token_count, 24)
    assert output.response_embedding.shape == (graph.response_count, 24)
    assert output.prefix_state.shape == output.response_embedding.shape
    assert output.lineage.shape == (
        graph.response_count,
        graph.layer_count,
        graph.head_count,
        3,
    )
    assert torch.equal(
        output.response_embedding,
        output.node_embedding[graph.response_start :],
    )
    assert torch.allclose(
        output.lineage.sum(dim=-1),
        torch.ones_like(output.lineage[..., 0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_full_encoding_equals_encoding_the_same_causal_prefix():
    torch.manual_seed(3)
    graph = make_graph()
    model = make_model(graph)
    full = model.encode(graph)

    for count in (1, 3, graph.response_count - 1):
        prefix = model.encode(graph.truncate_response(count))
        assert torch.allclose(
            full.response_embedding[:count],
            prefix.response_embedding,
            atol=1e-6,
            rtol=1e-5,
        )
        assert torch.allclose(
            full.prefix_state[:count],
            prefix.prefix_state,
            atol=1e-6,
            rtol=1e-5,
        )
        assert torch.allclose(
            full.lineage[:count],
            prefix.lineage,
            atol=1e-7,
            rtol=1e-6,
        )


def test_endpoint_predictor_cannot_read_current_or_future_target_rows():
    torch.manual_seed(5)
    graph = make_graph()
    model = make_model(graph)
    response_index = 3
    current = graph.response_start + response_index
    changed = change_current_and_future_rows(graph, current)
    current_or_future = graph.edges.target >= current
    assert not torch.equal(
        graph.edges.source[current_or_future],
        changed.edges.source[current_or_future],
    )
    assert torch.equal(graph.edges.weight, changed.edges.weight)

    original_output = model.encode(graph)
    changed_output = model.encode(changed)
    assert torch.allclose(
        original_output.response_embedding[:response_index],
        changed_output.response_embedding[:response_index],
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_output.lineage[:response_index],
        changed_output.lineage[:response_index],
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        original_output.prefix_state[: response_index + 1],
        changed_output.prefix_state[: response_index + 1],
        atol=1e-6,
        rtol=1e-5,
    )
    assert not torch.allclose(
        original_output.response_embedding[response_index],
        changed_output.response_embedding[response_index],
    )

    candidate_source = torch.tensor([0, current - 1])
    candidate_target = torch.full((2,), current, dtype=torch.long)
    candidate_layer = torch.tensor([0, graph.layer_count - 1])
    candidate_head = torch.tensor([0, graph.head_count - 1])
    original_score = model.endpoint_score(
        original_output,
        graph,
        candidate_source,
        candidate_target,
        candidate_layer,
        candidate_head,
    )
    changed_score = model.endpoint_score(
        changed_output,
        changed,
        candidate_source,
        candidate_target,
        candidate_layer,
        candidate_head,
    )
    assert torch.allclose(original_score, changed_score, atol=1e-6, rtol=1e-5)


def test_edge_storage_order_does_not_change_embeddings():
    torch.manual_seed(7)
    graph = make_graph()
    model = make_model(graph)
    order = torch.randperm(graph.edge_count, generator=torch.Generator().manual_seed(8))
    permuted = permute_edge_storage(graph, order)

    original = model.encode(graph)
    reordered = model.encode(permuted)
    assert torch.allclose(
        original.response_embedding,
        reordered.response_embedding,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(original.lineage, reordered.lineage, atol=1e-7, rtol=1e-6)


def test_matched_endpoint_change_affects_only_causal_descendants():
    torch.manual_seed(11)
    graph = make_rewirable_graph()
    rewired = rewire_endpoints_keep_roles(
        graph,
        torch.Generator().manual_seed(13),
        passes=1,
    )
    assert not torch.equal(graph.edges.source, rewired.edges.source)

    model = make_model(graph)
    original = model.encode(graph).response_embedding
    changed = model.encode(rewired).response_embedding

    first_changed_target = int(graph.edges.target.min()) - graph.response_start
    assert torch.allclose(
        original[:first_changed_target],
        changed[:first_changed_target],
        atol=1e-6,
        rtol=1e-5,
    )
    assert not torch.allclose(original[first_changed_target:], changed[first_changed_target:])


def test_route_negatives_match_role_lag_and_causality_without_observed_edges():
    graph = make_graph()
    pairs = matched_negative_edges(
        graph,
        count=4,
        generator=torch.Generator().manual_seed(15),
        attempt_factor=8,
        positive_edges_per_graph=3,
    )
    assert pairs.count > 0
    assert pairs.edge.unique().numel() <= 3
    assert int(pairs.edge.min()) >= 0
    assert int(pairs.edge.max()) < graph.edge_count

    edge = pairs.edge
    positive_source = graph.edges.source[edge]
    target = graph.edges.target[edge]
    assert torch.equal(
        positive_source >= graph.response_start,
        pairs.negative_source >= graph.response_start,
    )
    assert torch.equal(
        lag_bucket(target - positive_source, 63),
        lag_bucket(target - pairs.negative_source, 63),
    )
    assert bool((pairs.negative_source < target).all())

    observed = {
        (int(source), int(target), int(layer), int(head))
        for source, target, layer, head in zip(
            graph.edges.source.tolist(),
            graph.edges.target.tolist(),
            graph.edges.layer.tolist(),
            graph.edges.head.tolist(),
            strict=True,
        )
    }
    for pair, negative_source in zip(edge.tolist(), pairs.negative_source.tolist(), strict=True):
        candidate = (
            negative_source,
            int(graph.edges.target[pair]),
            int(graph.edges.layer[pair]),
            int(graph.edges.head[pair]),
        )
        assert candidate not in observed


def test_route_objective_is_finite_and_backpropagates():
    torch.manual_seed(17)
    graph = make_graph()
    model = make_model(graph).train()
    result = self_supervised_loss(
        model,
        graph,
        LearningConfig(
            positive_edges_per_graph=3,
            negative_count=4,
            negative_attempt_factor=8,
            variance_weight=0.05,
        ),
        torch.Generator().manual_seed(19),
    )

    assert result.pair_count > 0
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.route)
    assert torch.isfinite(result.variance)
    result.route.backward()
    route_gradients = [
        parameter.grad
        for parameter in (*model.route_query.parameters(), *model.route_key.parameters())
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert route_gradients
    assert any(bool((gradient.abs() > 0).any()) for gradient in route_gradients)
    transition_gradient = model.head_transition.logit.grad
    assert transition_gradient is not None
    assert bool((transition_gradient[1:].abs() > 0).any())


def test_training_forward_materializes_each_layer_once(monkeypatch):
    graph = make_graph()
    calls = []
    original = type(graph).layer_edges

    def tracked(self, layer, device=None):
        calls.append(layer)
        return original(self, layer, device)

    monkeypatch.setattr(type(graph), "layer_edges", tracked)
    make_model(graph).train().encode(graph)
    assert calls == list(range(graph.layer_count))


def test_cuda_keeps_full_edges_on_cpu_and_backpropagates_when_available():
    if not torch.cuda.is_available():
        return

    graph = make_graph().to("cuda")
    model = make_model(graph).to("cuda").train()
    result = self_supervised_loss(
        model,
        graph,
        LearningConfig(
            positive_edges_per_graph=3,
            negative_count=2,
            negative_attempt_factor=4,
            variance_weight=0.05,
        ),
        torch.Generator(device="cuda").manual_seed(37),
    )
    assert graph.edges.source.device.type == "cpu"
    assert result.loss.device.type == "cuda"
    result.loss.backward()
    assert model.route_key[1].weight.grad is not None
