from types import SimpleNamespace

import torch

from experiments.reanchor_flow.routes import RouteAccumulator


class DummyModel:
    config = SimpleNamespace(num_hidden_layers=3)


def test_accumulator_builds_attention_and_exact_output_projected_maps():
    accumulator = RouteAccumulator(DummyModel(), response_start=3, query_chunk=2)
    probability = torch.zeros(1, 2, 5, 5)
    probability[:, :, :, 0] = 1.0
    value = torch.zeros(1, 2, 5, 2)
    value[:, 0, 0] = torch.tensor([3.0, 4.0])
    value[:, 1, 0] = torch.tensor([0.0, 2.0])
    output = torch.eye(4)

    for layer in range(3):
        accumulator.observe(layer, probability, value, output)
    result = accumulator.finish()

    assert result.functional.shape == (3, 5)
    torch.testing.assert_close(result.functional[:, 0], torch.full((3,), 3.5))
    torch.testing.assert_close(result.attention[:, 0], torch.ones(3))
    torch.testing.assert_close(result.functional_middle, result.functional)
