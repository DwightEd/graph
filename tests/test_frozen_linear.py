import torch
from experiments.decision_risk_flow.precision import FrozenLinear


def test_frozen_linear_matches_fp32_forward_and_input_gradient():
    torch.manual_seed(7)
    weights = torch.randn(7, 9).bfloat16()
    inputs = torch.randn(3, 9, requires_grad=True)
    direction = torch.randn(3, 7)
    expected = torch.nn.functional.linear(inputs, weights.float())
    actual = FrozenLinear.apply(inputs, weights)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad((actual * direction).sum(), inputs)[0]
    reference = torch.autograd.grad((expected * direction).sum(), inputs)[0]
    torch.testing.assert_close(gradient, reference)
