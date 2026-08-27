import torch

from experiments.directed_route_hypergraph.posterior import (
    VariationalRoutePosterior,
)


def test_mean_logvar_export_doubles_dimension_and_eval_is_deterministic():
    torch.manual_seed(101)
    posterior = VariationalRoutePosterior(
        16,
        export="mean_logvar",
        logvar_min=-8.0,
        logvar_max=4.0,
    ).eval()
    state = torch.randn(7, 16)

    first = posterior(state)
    second = posterior(state)

    assert first.decoder_embedding.shape == (7, 16)
    assert first.exported_embedding.shape == (7, 32)
    assert torch.equal(first.decoder_embedding, first.mean)
    assert torch.equal(first.exported_embedding[:, :16], first.mean)
    assert torch.equal(first.exported_embedding[:, 16:], first.log_variance)
    assert torch.equal(first.decoder_embedding, second.decoder_embedding)
    assert torch.equal(first.exported_embedding, second.exported_embedding)


def test_mean_export_keeps_the_latent_dimension():
    posterior = VariationalRoutePosterior(
        12,
        export="mean",
        logvar_min=-8.0,
        logvar_max=4.0,
    ).eval()
    output = posterior(torch.randn(5, 12))

    assert output.decoder_embedding.shape == (5, 12)
    assert output.exported_embedding.shape == (5, 12)
    assert torch.equal(output.exported_embedding, output.mean)
