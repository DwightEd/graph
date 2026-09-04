from types import SimpleNamespace

import pytest
import torch

from experiments.reanchor_flow.routes import RouteAccumulator


class DummyModel:
    config = SimpleNamespace(num_hidden_layers=3)


def test_source_norm_matches_direct_head_projection():
    torch.manual_seed(7)
    heads, sources, head_dim, hidden = 4, 6, 3, 12
    value = torch.randn(heads, sources, head_dim)
    output = torch.randn(hidden, heads * head_dim)

    actual = RouteAccumulator.source_norm(value, output)
    blocks = output.reshape(hidden, heads, head_dim).permute(1, 2, 0)
    expected = torch.stack(
        [torch.linalg.vector_norm(value[h] @ blocks[h], dim=-1) for h in range(heads)]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_accumulator_keeps_layer_and_role_coordinates():
    accumulator = RouteAccumulator(
        DummyModel(),
        response_start=3,
        prompt_evidence_mask=[True, False, False],
    )
    probability = torch.zeros(1, 2, 5, 5)
    probability[:, :, :, 0] = 1.0
    value = torch.zeros(1, 2, 5, 2)
    value[:, 0, 0] = torch.tensor([3.0, 4.0])
    value[:, 1, 0] = torch.tensor([0.0, 2.0])
    output = torch.eye(4)

    for layer in range(3):
        accumulator.observe(layer, probability, value, output)
    result = accumulator.finish()

    assert result.functional_share.shape == (3, 3, 3)
    torch.testing.assert_close(
        result.functional_share[:, :, 0], torch.ones(3, 3)
    )
    torch.testing.assert_close(
        result.attention_share[:, :, 0], torch.ones(3, 3)
    )
    torch.testing.assert_close(
        result.functional_mass, torch.full((3, 3), 3.5)
    )
    torch.testing.assert_close(
        result.functional_null[:, :, 0], torch.ones(3, 3)
    )
    expected_attention_null = torch.tensor(
        [[1 / 3, 2 / 3, 0], [1 / 4, 2 / 4, 1 / 4], [1 / 5, 2 / 5, 2 / 5]]
    )
    torch.testing.assert_close(
        result.attention_null,
        expected_attention_null.expand(3, -1, -1),
    )


def test_chunk_coordinates_fill_each_response_event_once():
    accumulator = RouteAccumulator(
        DummyModel(),
        response_start=3,
        prompt_evidence_mask=[True, False, False],
    )
    probability = torch.zeros(1, 2, 2, 5)
    probability[:, :, :, 0] = 1.0
    value = torch.ones(1, 2, 5, 2)
    output = torch.eye(4)

    for layer in range(3):
        accumulator.observe_chunk(layer, 0, probability, value, output)
        accumulator.observe_chunk(layer, 2, probability, value, output)
        accumulator.observe_chunk(
            layer, 4, probability[:, :, :1], value, output
        )
    result = accumulator.finish()
    assert result.functional_share.shape == (3, 3, 3)
    torch.testing.assert_close(result.functional_null, result.attention_null)


def test_chunk_coordinates_reject_overlap():
    accumulator = RouteAccumulator(
        DummyModel(),
        response_start=3,
        prompt_evidence_mask=[True, False, False],
    )
    probability = torch.zeros(1, 2, 2, 5)
    value = torch.ones(1, 2, 5, 2)
    output = torch.eye(4)

    accumulator.observe_chunk(0, 0, probability, value, output)
    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        accumulator.observe_chunk(0, 0, probability, value, output)


def test_chunk_coordinates_reject_gap():
    accumulator = RouteAccumulator(
        DummyModel(),
        response_start=3,
        prompt_evidence_mask=[True, False, False],
    )
    probability = torch.zeros(1, 2, 1, 5)
    value = torch.ones(1, 2, 5, 2)
    output = torch.eye(4)

    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        accumulator.observe_chunk(0, 1, probability, value, output)
