import torch
from unittest.mock import patch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.decision_risk_flow.native import prefill, replay
from experiments.decision_risk_flow.precision import enable_fp32_execution, FrozenLinear
from experiments.message_js.operator import response_tangents, full_vocabulary_metric
from experiments.decision_risk_flow.kernel import contract_edges
from experiments.message_js.history_trace import full_trace
from experiments.message_js.signed_score import signed_scores
import numpy as np


def test_frozen_weight_double_backward_matches_jvp():
    torch.manual_seed(21)
    weight = torch.randn(7, 9).bfloat16()
    value, direction = torch.randn(3, 9), torch.randn(3, 9)
    _, tangent = torch.autograd.functional.jvp(lambda x: FrozenLinear.apply(x, weight), value, direction)
    torch.testing.assert_close(tangent, torch.nn.functional.linear(direction, weight.float()))


def test_native_jvp_full_distribution_and_zero_gate_identity():
    torch.manual_seed(22)
    config = LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = enable_fp32_execution(LlamaForCausalLM(config).bfloat16().eval().requires_grad_(False))
    prompt, answer = [1, 2, 3], [4, 5, 6]
    tokens = prompt + answer[:-1]
    cache, _, base = prefill(model, prompt, answer, checkpoints=(2,))
    positions = torch.tensor([2, 3, 4])
    final, tangent = response_tangents(model, cache, tokens, positions, len(prompt))
    torch.testing.assert_close(final, base, atol=2e-6, rtol=1e-5)
    for channel in range(3):
        direction = torch.eye(3)[channel]
        changed = []
        for sign in (-1, 1):
            value, _, _ = replay(model, cache, tokens, positions, (2,),
                gate=dict(scales=torch.ones(3) + sign * .005 * direction, prompt_length=len(prompt)))
            changed.append(value.detach())
        finite = (changed[1] - changed[0]) / .01
        torch.testing.assert_close(finite, tangent[:, channel], atol=2e-4, rtol=.01)
    gram, _ = full_vocabulary_metric(model, final, tangent, torch.tensor(answer), torch.tensor([7,8,9]))
    assert torch.linalg.eigvalsh(gram).min() > -1e-7
    # Same output alphabet: check the 1/8 JS curvature coefficient independently.
    epsilon = 1e-3
    baseline_logits = model.lm_head(final).double()
    response = model.lm_head(tangent).double()
    logp = [(baseline_logits + epsilon * response[:, channel]).log_softmax(-1) for channel in (0, 1)]
    mixture = torch.logaddexp(logp[0], logp[1]) - torch.log(torch.tensor(2., dtype=torch.float64))
    divergence = .5 * sum((value.exp() * (value - mixture)).sum(-1) for value in logp)
    expected = (gram[:, 0, 0] + gram[:, 1, 1] - 2 * gram[:, 0, 1]) / 8
    torch.testing.assert_close(divergence / epsilon**2, expected.double(), atol=1e-7, rtol=.005)
    # The first prediction has no response-history messages.
    torch.testing.assert_close(tangent[0, 1], torch.zeros_like(tangent[0, 1]))


def test_residual_component_matches_native_ffn_jacobian_bypass():
    torch.manual_seed(23)
    config = LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    prompt, answer = [1, 2, 3], [4, 5]
    tokens = prompt + answer[:-1]
    cache, _, _ = prefill(model, prompt, answer, checkpoints=(2,))
    position = torch.tensor([3])
    final, _, capture = replay(model, cache, tokens, position, (2,))
    objective = model.lm_head(final)[0, 5]
    gradient = torch.autograd.grad(objective, capture.mlp_writes[0])[0]
    residual = contract_edges(gradient, capture.messages[0], model.model.layers[0].self_attn.o_proj.weight)[0, 0, 0]
    mlp = model.model.layers[0].mlp
    original = mlp.forward
    scale = torch.tensor(1., requires_grad=True)
    with patch.object(mlp, 'forward', lambda value: original(value).detach()):
        changed, _, _ = replay(model, cache, tokens, position, (2,),
                              gate=dict(layer=0, head=0, key=0, scale=scale))
    torch.testing.assert_close(final, changed)
    actual = torch.autograd.grad(model.lm_head(changed)[0, 5], scale)[0]
    torch.testing.assert_close(actual, residual, atol=1e-7, rtol=1e-5)


def test_true_history_trace_obeys_causality_and_last_layer_boundary():
    torch.manual_seed(24)
    config = LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    tokens = [1, 2, 3, 4, 5, 6]
    for layer in (0, 2):
        _, response, changed = full_trace(model, tokens, layer, 0, 3, 0)
        torch.testing.assert_close(response[:3], torch.zeros_like(response[:3]))
        assert response[3].norm() > 1e-6
        if layer == 2:
            torch.testing.assert_close(response[4:], torch.zeros_like(response[4:]))
        else:
            assert response[4:].norm() > 1e-7


def test_signed_fisher_score_is_invariant_to_positive_direction_rescaling():
    margin = np.array([[2., -3., 1.], [1., 2., 0.]])
    gram = np.array([[[4., 1., 0.], [1., 9., 0.], [0., 0., 1.]],
                     [[0., 0., 0.], [0., 4., 0.], [0., 0., 1.]]])
    scale = np.array([7., 2., 3.])
    original = signed_scores(margin, gram)
    changed = signed_scores(margin * scale, gram * scale[None, :, None] * scale[None, None, :])
    for name in ('signed_prompt_contrast', 'history_over_prompt'):
        np.testing.assert_allclose(original[name], changed[name], equal_nan=True)
        assert np.isnan(original[name][1])
    assert original['signed_prompt_contrast'][0] < 0
