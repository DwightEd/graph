from dataclasses import replace

import torch

from experiments.grounded_route.graph import build_graph, endpoint_storage_keys
from experiments.grounded_route.tests.helpers import SparseSample, make_graph


def retained_mass(graph):
    mass = torch.zeros_like(graph.diagonal)
    mass.index_put_(
        (
            graph.edges.target - graph.response_start,
            graph.edges.layer,
            graph.edges.head,
        ),
        graph.edges.weight,
        accumulate=True,
    )
    return mass


def test_sparse_csr_keeps_all_1024_layer_head_channels():
    sample = SparseSample(layers=32, heads=32)
    graph = build_graph(sample)

    assert graph.diagonal.shape == (sample.response_count, 32, 32)
    assert graph.unresolved.shape == graph.diagonal.shape
    assert graph.edge_count == sample.response_count * 32 * 32

    for response_index in range(sample.response_count):
        selected = graph.edges.target == graph.response_start + response_index
        channel = graph.edges.layer[selected] * graph.head_count + graph.edges.head[selected]
        assert torch.equal(torch.sort(channel).values, torch.arange(32 * 32))

    last_channel = (graph.edges.layer == 31) & (graph.edges.head == 31)
    assert torch.allclose(
        graph.edges.weight[last_channel],
        torch.full((sample.response_count,), 0.10 + 1023e-6),
    )


def test_retained_diagonal_and_censored_mass_are_conserved_per_row():
    graph = build_graph(SparseSample())
    retained = retained_mass(graph)

    assert torch.all(graph.unresolved > 0)
    assert torch.allclose(
        retained + graph.diagonal + graph.unresolved,
        torch.ones_like(graph.diagonal),
        atol=1e-7,
        rtol=0.0,
    )


def test_response_truncation_is_an_exact_causal_subgraph():
    graph = make_graph()
    prefix = graph.truncate_response(4)
    stop = graph.response_start + 4

    assert prefix.response_count == 4
    assert prefix.token_count == stop
    assert torch.equal(prefix.token_ids, graph.token_ids[:stop])
    assert torch.equal(prefix.edges.source, graph.edges.source[graph.edges.target < stop])
    assert torch.equal(prefix.edges.target, graph.edges.target[graph.edges.target < stop])
    assert bool((prefix.edges.source < prefix.edges.target).all())


def test_graph_canonicalizes_once_and_exposes_contiguous_cpu_layer_slices():
    graph = make_graph()
    order = torch.randperm(graph.edge_count, generator=torch.Generator().manual_seed(29))
    edges = graph.edges.select(order)
    checked = replace(graph, edges=edges).check()
    assert checked.edges is edges
    graph = checked.canonicalize()

    key = endpoint_storage_keys(graph)
    assert graph.edges.source.device.type == "cpu"
    assert bool((key[1:] >= key[:-1]).all())
    assert graph.layer_offsets[0] == 0
    assert graph.layer_offsets[-1] == graph.edge_count
    for layer in range(graph.layer_count):
        selected = graph.layer_edges(layer)
        assert bool((selected.layer == layer).all())


def test_to_keeps_sparse_topology_on_cpu():
    graph = make_graph().to("cpu")
    assert graph.edges.source.device.type == "cpu"
    assert graph.edges.weight.device.type == "cpu"
    assert graph.token_ids.device.type == "cpu"
