from dataclasses import replace

import torch

from experiments.directed_route_hypergraph.config import ModelConfig
from experiments.directed_route_hypergraph.corruption import corrupt_graph
from experiments.directed_route_hypergraph.model import DirectedRouteHypergraphEncoder
from experiments.grounded_route.tests.helpers import make_graph, permute_edge_storage


def make_model(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(dropout=0.0),
    )


def without_native_unresolved(graph):
    retained = torch.zeros_like(graph.unresolved)
    retained.index_put_(
        (
            graph.edges.target - graph.response_start,
            graph.edges.layer,
            graph.edges.head,
        ),
        graph.edges.weight,
        accumulate=True,
    )
    return replace(
        graph,
        diagonal=1.0 - retained,
        unresolved=torch.zeros_like(graph.unresolved),
    ).check()


def test_deterministic_encoder_exports_64d_final_state_without_layer_inputs():
    torch.manual_seed(3)
    graph = make_graph()
    model = make_model(graph).train()
    output = model(graph)

    assert output.node_embedding.shape == (graph.token_count, 64)
    assert output.response_embedding.shape == (graph.response_count, 64)
    assert output.decoder_embedding.shape == (graph.token_count, 64)
    assert output.decoder_response_embedding.shape == (graph.response_count, 64)
    assert output.posterior_mean.shape == (graph.token_count, 64)
    assert torch.equal(
        output.posterior_log_variance,
        torch.zeros_like(output.posterior_log_variance),
    )
    assert not hasattr(output, "layer_input")
    assert output.flow_logits.shape == (
        graph.response_count,
        graph.layer_count,
        3,
    )
    assert torch.allclose(
        output.lineage.sum(dim=-1),
        torch.ones_like(output.lineage[..., 0]),
        atol=1e-6,
        rtol=1e-6,
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


def test_endpoint_score_reads_final_decoder_embedding_not_export_aliases():
    torch.manual_seed(13)
    graph = make_graph()
    model = make_model(graph).eval()
    output = model(graph)
    edge = torch.tensor([0, graph.edge_count // 2], dtype=torch.long)
    source = graph.edges.source[edge]
    target = graph.edges.target[edge]
    layer = graph.edges.layer[edge]
    head = graph.edges.head[edge]
    baseline = model.endpoint_score(output, graph, source, target, layer, head)

    exported_only = replace(
        output,
        node_embedding=torch.randn_like(output.node_embedding),
        response_embedding=torch.randn_like(output.response_embedding),
    )
    assert torch.allclose(
        model.endpoint_score(exported_only, graph, source, target, layer, head),
        baseline,
    )

    decoder = output.decoder_embedding.clone()
    decoder[target] += torch.linspace(-1.0, 1.0, decoder.shape[1])
    final_state_changed = replace(output, decoder_embedding=decoder)
    assert not torch.allclose(
        model.endpoint_score(
            final_state_changed,
            graph,
            source,
            target,
            layer,
            head,
        ),
        baseline,
    )


def test_artificial_mask_and_native_unresolved_use_distinct_messages():
    torch.manual_seed(17)
    graph = without_native_unresolved(make_graph(layers=1, heads=1))
    result = corrupt_graph(
        graph,
        incidence_dropout=0.0,
        head_dropout=0.0,
        generator=torch.Generator().manual_seed(19),
        forced_edge=torch.tensor([0]),
    )
    masked_model = make_model(graph).train()
    masked_output = masked_model(result.graph, masked_mass=result.masked_mass)
    masked_output.decoder_response_embedding.square().sum().backward()

    masked_gradient = masked_model.source_to_hyperedge.masked_message.grad
    unresolved_gradient = masked_model.source_to_hyperedge.unresolved_message.grad
    assert masked_gradient is not None and bool(masked_gradient.abs().sum() > 0)
    assert unresolved_gradient is not None
    assert torch.equal(unresolved_gradient, torch.zeros_like(unresolved_gradient))

    native_graph = make_graph(layers=1, heads=1)
    unresolved_model = make_model(native_graph).train()
    unresolved_output = unresolved_model(native_graph)
    unresolved_output.decoder_response_embedding.square().sum().backward()

    native_gradient = unresolved_model.source_to_hyperedge.unresolved_message.grad
    artificial_gradient = unresolved_model.source_to_hyperedge.masked_message.grad
    assert native_gradient is not None and bool(native_gradient.abs().sum() > 0)
    assert artificial_gradient is not None
    assert torch.equal(artificial_gradient, torch.zeros_like(artificial_gradient))
