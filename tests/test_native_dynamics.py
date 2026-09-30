"""Mathematical checks, not natural hallucination-detection experiments."""

from types import SimpleNamespace

import numpy as np
import torch
from state_audit.analysis.ffn_transport import ffn_transport


def test_ffn_tangent_matches_autograd_and_small_directional_difference():
    random = torch.Generator().manual_seed(17)
    def linear(inputs, outputs):
        layer = torch.nn.Linear(inputs, outputs, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(outputs, inputs, generator=random) / inputs**.5)
        return layer

    block = SimpleNamespace(
        post_attention_layernorm=SimpleNamespace(
            weight=torch.linspace(.5, 1.5, 5, dtype=torch.float64), variance_epsilon=1e-6,
        ),
        mlp=SimpleNamespace(gate_proj=linear(5, 9), up_proj=linear(5, 9), down_proj=linear(9, 5)),
    )
    residual = torch.randn(5, dtype=torch.float64, generator=random)
    directions = torch.randn(3, 5, dtype=torch.float64, generator=random)
    def native(value):
        normalized = block.post_attention_layernorm.weight * value / (value.square().mean() + 1e-6).sqrt()
        return value + block.mlp.down_proj(
            torch.nn.functional.silu(block.mlp.gate_proj(normalized)) * block.mlp.up_proj(normalized)
        )

    output, tangents = ffn_transport(block, residual, directions)
    jacobian = torch.autograd.functional.jacobian(native, residual)
    torch.testing.assert_close(output, native(residual))
    torch.testing.assert_close(tangents, directions @ jacobian.T)
    step = 1e-5
    for direction, tangent in zip(directions, tangents):
        difference = (native(residual + step * direction) - native(residual - step * direction)) / (2 * step)
        torch.testing.assert_close(tangent, difference, atol=1e-8, rtol=1e-7)


def test_reverse_write_can_preserve_decision_information_by_rotation():
    evidence = np.array([1., 0.])
    transform = np.array([[0., -1.], [1., 0.]])
    ff_write = transform @ evidence - evidence
    output_direction = np.array([0., 1.])
    assert ff_write @ evidence < 0
    assert output_direction @ transform @ evidence == 1


def test_input_to_output_transfer_is_invariant_to_latent_coordinates():
    transition = np.array([[.9, .2], [.1, .8]])
    injection = np.array([[1.], [.3]])
    readout = np.array([[2., -1.]])
    coordinates = np.array([[2., .5], [.2, 1.]])
    inverse = np.linalg.inv(coordinates)
    expected = readout @ transition @ transition @ injection
    changed = readout @ inverse @ np.linalg.matrix_power(coordinates @ transition @ inverse, 2) @ coordinates @ injection
    np.testing.assert_allclose(changed, expected)
