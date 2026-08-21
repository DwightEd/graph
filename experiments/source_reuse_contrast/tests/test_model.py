import torch

from experiments.source_reuse_contrast.data import collect_source_reuse_graph
from experiments.source_reuse_contrast.model import SourceReusePredictor

from .helpers import sequence_sample, tiny_config


def test_info_nce_scores_are_raw_bounded_and_have_gradients():
    graph = collect_source_reuse_graph(sequence_sample())
    config = tiny_config()
    model = SourceReusePredictor(
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        config=config,
    )
    output = model(graph, seed=19)
    loss = model.loss(output)
    loss.backward()

    assert bool(output.valid.any())
    assert torch.isfinite(loss)
    bound = 1.0 / config.temperature + 1e-5
    assert output.positive_logit.abs().max() <= bound
    assert output.hardest_negative_logit.abs().max() <= bound
    assert output.endpoint_nll[output.valid].unique().numel() > 1
    assert sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    ) > 0.0


def test_prefix_scores_do_not_depend_on_final_response_length():
    short = collect_source_reuse_graph(sequence_sample(sample_id="short"))
    long = collect_source_reuse_graph(
        sequence_sample(sample_id="long", extra_token=True)
    )
    model = SourceReusePredictor(
        num_layers=2,
        num_heads=2,
        config=tiny_config(),
    ).eval()

    with torch.no_grad():
        short_output = model(short, seed=23)
        long_output = model(long, seed=23)

    torch.testing.assert_close(
        short_output.query_embedding,
        long_output.query_embedding[: short.num_response_tokens],
        atol=1e-6,
        rtol=1e-6,
    )
