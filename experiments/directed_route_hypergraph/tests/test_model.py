from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.config import ModelConfig
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.graph import TokenEdges
from experiments.grounded_route.tests.helpers import make_graph, permute_edge_storage


def make_model(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    )


def change_one_row(graph, response_index: int, layer: int, head: int):
    target = graph.response_start + response_index
    selected = (
        (graph.edges.target == target)
        & (graph.edges.layer == layer)
        & (graph.edges.head == head)
    )
    edge = int(torch.nonzero(selected, as_tuple=False)[0].item())
    delta = 0.02
    weight = graph.edges.weight.clone()
    weight[edge] += delta
    unresolved = graph.unresolved.clone()
    unresolved[response_index, layer, head] -= delta
    return replace(
        graph,
        edges=TokenEdges(
            source=graph.edges.source,
            target=graph.edges.target,
            layer=graph.edges.layer,
            head=graph.edges.head,
            weight=weight,
        ),
        unresolved=unresolved,
    ).check()


def test_encoder_returns_64_dimensions_and_both_message_stages_receive_gradient():
    torch.manual_seed(3)
    graph = make_graph()
    model = make_model(graph).train()
    output = model(graph, return_layer_input=True)

    assert output.node_embedding.shape == (graph.token_count, 64)
    assert output.response_embedding.shape == (graph.response_count, 64)
    assert output.layer_input.shape == (
        graph.layer_count,
        graph.token_count,
        4,
        16,
    )

    coefficient = torch.arange(1, 65, dtype=output.response_embedding.dtype)
    (output.response_embedding * coefficient).sum().backward()
    source_gradient = model.source_to_hyperedge.source_projection[0][-1].weight.grad
    target_gradient = model.hyperedge_to_target.update[0].weight_ih.grad
    assert source_gradient is not None and bool(source_gradient.abs().sum() > 0)
    assert target_gradient is not None and bool(target_gradient.abs().sum() > 0)


def test_edge_storage_permutation_does_not_change_node_embeddings():
    torch.manual_seed(5)
    graph = make_graph()
    order = torch.randperm(graph.edge_count, generator=torch.Generator().manual_seed(7))
    permuted = permute_edge_storage(graph, order)
    model = make_model(graph).eval()

    original = model(graph)
    reordered = model(permuted)
    assert torch.allclose(
        original.node_embedding,
        reordered.node_embedding,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(original.lineage, reordered.lineage, atol=1e-7, rtol=1e-6)


def test_full_graph_and_truncated_graph_have_identical_prefix_embeddings():
    torch.manual_seed(11)
    graph = make_graph()
    model = make_model(graph).eval()
    full = model(graph)

    for count in (1, 3, graph.response_count - 1):
        prefix = model(graph.truncate_response(count))
        assert torch.allclose(
            full.response_embedding[:count],
            prefix.response_embedding,
            atol=1e-6,
            rtol=1e-5,
        )
        assert torch.allclose(
            full.lineage[:count],
            prefix.lineage,
            atol=1e-7,
            rtol=1e-6,
        )


def test_current_row_changes_post_update_but_not_pre_consume_scores():
    torch.manual_seed(13)
    graph = make_graph()
    response_index, layer, head = 3, 0, 0
    target = graph.response_start + response_index
    changed = change_one_row(graph, response_index, layer, head)
    model = make_model(graph).eval()

    original = model(graph, return_layer_input=True)
    modified = model(changed, return_layer_input=True)
    candidate_source = torch.tensor([0, target - 1])
    candidate_target = torch.full((2,), target, dtype=torch.long)
    candidate_layer = torch.full((2,), layer, dtype=torch.long)
    candidate_head = torch.full((2,), head, dtype=torch.long)
    original_score = model.endpoint_score(
        original,
        graph,
        candidate_source,
        candidate_target,
        candidate_layer,
        candidate_head,
    )
    changed_score = model.endpoint_score(
        modified,
        changed,
        candidate_source,
        candidate_target,
        candidate_layer,
        candidate_head,
    )

    assert torch.allclose(original_score, changed_score, atol=1e-6, rtol=1e-5)
    assert not torch.allclose(
        original.response_embedding[response_index],
        modified.response_embedding[response_index],
    )
