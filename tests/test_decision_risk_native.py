"""Check timestep isolation and native downstream derivatives before 8B runs."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.decision_risk_flow.native import prefill, replay


def test_independent_queries_match_native_and_do_not_read_each_other():
    torch.manual_seed(4)
    config = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    prompt, answer = [1, 2, 3, 4], [5, 6, 7]
    cache, raw, final = prefill(model, prompt, answer, checkpoints=(1, 2), chunk=3)
    positions = torch.tensor([3, 4, 5])
    replayed, states, capture = replay(model, cache, prompt + answer[:-1], positions, (1, 2))
    torch.testing.assert_close(replayed, final, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(states, raw, atol=2e-6, rtol=2e-5)
    for i in range(3):
        single, _, _ = replay(model, cache, prompt + answer[:-1], positions[i:i+1], (1, 2))
        torch.testing.assert_close(replayed[i], single[0], atol=2e-6, rtol=2e-5)
    derivative = torch.autograd.grad(replayed[0].sum(), capture.writes[0])[0]
    torch.testing.assert_close(derivative[0, 1:], torch.zeros_like(derivative[0, 1:]))


def test_message_gradient_matches_explicit_gate_through_downstream_layers():
    torch.manual_seed(8)
    config = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).double().eval().requires_grad_(False)
    prompt, answer = [1, 2, 3], [4, 5]
    cache, _, _ = prefill(model, prompt, answer, checkpoints=(1, 2))
    positions = torch.tensor([2, 3])
    final, _, capture = replay(model, cache, prompt + answer[:-1], positions, (1, 2))
    score = model.lm_head(final)[:, 4].sum()
    gradient = torch.autograd.grad(score, capture.writes[0])[0][0]
    weights, values, _ = capture.messages[0]
    head_weight = model.model.layers[0].self_attn.o_proj.weight[:, :4]
    messages = weights[0, :, 0, None] * (head_weight @ values[0, 0])[None]
    expected = (gradient * messages).sum()
    scale = torch.tensor(1., dtype=torch.double, requires_grad=True)
    final, _, _ = replay(model, cache, prompt + answer[:-1], positions, (1, 2),
        dict(layer=0, head=0, key=0, scale=scale))
    actual = torch.autograd.grad(model.lm_head(final)[:, 4].sum(), scale)[0]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
