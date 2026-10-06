import torch
from experiments.decision_risk_flow.precision import FrozenLinear, chunk_unembedding


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


def test_tiled_vocabulary_preserves_all_logits_and_input_derivative():
    from types import SimpleNamespace

    torch.manual_seed(17)
    head = torch.nn.Linear(24, 103, bias=False).bfloat16().requires_grad_(False)
    model = SimpleNamespace(lm_head=head)
    inputs = torch.randn(2, 3, 24, requires_grad=True)
    direction = torch.randn(2, 3, 103)
    expected = torch.nn.functional.linear(inputs, head.weight.float())
    reference = torch.autograd.grad((expected * direction).sum(), inputs, create_graph=True)[0]
    chunk_unembedding(model, rows=32)
    actual = head(inputs)
    gradient = torch.autograd.grad((actual * direction).sum(), inputs, create_graph=True)[0]
    assert actual.shape[-1] == 103  # Includes the non-multiple tail, no vocabulary screening.
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient, reference)
