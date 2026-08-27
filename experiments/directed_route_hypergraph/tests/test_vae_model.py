import torch

from experiments.directed_route_hypergraph.config import ModelConfig
from experiments.directed_route_hypergraph.model import (
    DirectedRouteHypergraphEncoder,
)
from experiments.grounded_route.tests.helpers import make_graph


def make_vae(graph):
    return DirectedRouteHypergraphEncoder(
        graph.layer_count,
        graph.head_count,
        ModelConfig(
            dropout=0.0,
            latent_mode="vae",
            vae_export="mean_logvar",
        ),
    )


def test_full_vae_encoder_exports_mean_logvar_without_stochastic_eval_noise():
    torch.manual_seed(107)
    graph = make_graph()
    model = make_vae(graph).eval()

    first = model.encode(graph)
    second = model.encode(graph)
    hidden = model.config.hidden_dim

    assert first.decoder_embedding.shape == (graph.token_count, hidden)
    assert first.node_embedding.shape == (graph.token_count, 2 * hidden)
    assert first.response_embedding.shape == (graph.response_count, 2 * hidden)
    assert torch.equal(first.node_embedding[:, :hidden], first.posterior_mean)
    assert torch.equal(
        first.node_embedding[:, hidden:],
        first.posterior_log_variance,
    )
    assert torch.equal(first.decoder_embedding, first.posterior_mean)
    assert torch.equal(first.node_embedding, second.node_embedding)
    assert torch.equal(first.decoder_embedding, second.decoder_embedding)


def test_endpoint_decoder_backpropagates_through_sampled_vae_latent():
    torch.manual_seed(109)
    graph = make_graph()
    model = make_vae(graph).train()
    output = model(graph)
    selected = torch.arange(min(4, graph.edge_count))

    score = model.endpoint_score(
        output,
        graph,
        graph.edges.source[selected],
        graph.edges.target[selected],
        graph.edges.layer[selected],
        graph.edges.head[selected],
    )
    score.sum().backward()

    assert model.posterior is not None
    mean_gradient = model.posterior.mean.weight.grad
    logvar_gradient = model.posterior.log_variance.weight.grad
    assert mean_gradient is not None and bool(mean_gradient.abs().sum() > 0)
    assert logvar_gradient is not None and bool(logvar_gradient.abs().sum() > 0)
