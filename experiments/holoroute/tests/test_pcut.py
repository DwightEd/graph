import torch

from experiments.holoroute.config import PCutConfig
from experiments.holoroute.pcut import (
    build_views,
    compute_pcut,
    prompt_provenance,
    row_index,
    split_edges,
)
from experiments.holoroute.tests.helpers import synthetic_graph


def test_provenance_bounds_and_edge_partition():
    graph = synthetic_graph()
    provenance = prompt_provenance(graph)
    assert provenance.lower_after.shape == (3, 3, 2)
    assert torch.all(provenance.lower_after <= provenance.upper_after + 1e-7)
    assert torch.all((provenance.lower_after >= 0) & (provenance.upper_after <= 1))

    parts = split_edges(graph, provenance)
    reconstructed = parts.prompt_rooted + parts.response_closed + parts.uncertain
    assert torch.allclose(reconstructed, graph.edges.weight, atol=1e-6)


def test_mass_preserving_cuts():
    graph = synthetic_graph()
    config = PCutConfig(identity_dim=8, head_projection_dim=2, tail_layers=2)
    parts = split_edges(graph, prompt_provenance(graph))
    views = build_views(graph, parts, config)
    index = row_index(graph)
    rows = graph.response_count * graph.layer_count * graph.head_count

    original = torch.zeros(rows)
    no_prompt = torch.zeros(rows)
    no_closed = torch.zeros(rows)
    original.index_add_(0, index, views.full)
    no_prompt.index_add_(0, index, views.no_prompt)
    no_closed.index_add_(0, index, views.no_closed)

    prompt_supported = views.no_prompt_supported.reshape(-1)
    closed_supported = views.no_closed_supported.reshape(-1)
    assert torch.allclose(original[prompt_supported], no_prompt[prompt_supported], atol=1e-6)
    assert torch.allclose(original[closed_supported], no_closed[closed_supported], atol=1e-6)


def test_pcut_exports_token_embeddings_and_single_score():
    graph = synthetic_graph()
    config = PCutConfig(identity_dim=8, head_projection_dim=2, tail_layers=2)
    result = compute_pcut(graph, config)
    assert result.token_layer_embedding.shape == (3, 3, 16)
    assert result.token_embedding.shape == (3, 16)
    assert result.closure.shape == (3,)
    assert torch.isfinite(result.closure).all()
    assert torch.allclose(
        result.closure,
        result.response_closed_necessity - result.prompt_necessity,
    )
