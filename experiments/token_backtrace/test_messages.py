"""Scientific contracts for native whole-head tracing, alignment and finite patches."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .messages import (aligned_deltas, capture_messages, direction_vector,
                       directional_vjp, finite_effect, native_trace,
                       patch_message, top_heads)


@pytest.fixture(params=('eager', 'sdpa'))
def model(request):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=3, num_attention_heads=3,
                        num_key_value_heads=1, attention_dropout=0.,
                        _attn_implementation=request.param)
    return LlamaForCausalLM(config).double().eval().requires_grad_(False)


def worlds(model):
    prompt, answer = [1, 2, 3], [4, 5, 6, 7, 8]
    original = native_trace(model, prompt, answer)
    donors = {name: native_trace(model, prompt, [token, *answer[1:]], gradients=False)
              for name, token in zip(('R', 'E', 'U'), (9, 10, 11))}
    alignments = {name: [-1, 1, 2, 3, 4] for name in donors}
    return original, donors, alignments


def test_prediction_and_carrier_ports_have_the_causal_shift(model):
    original, donors, _ = worlds(model)
    native = model(torch.tensor([[*original.prompt, *original.answer]])).logits[0]
    torch.testing.assert_close(original.logits, native[original.prediction_positions])
    assert original.carrier_positions.tolist() == [3, 4, 5, 6, 7]
    assert original.prediction_positions.tolist() == [2, 3, 4, 5, 6]
    torch.testing.assert_close(original.logits[0], donors['R'].logits[0], rtol=0, atol=0)
    assert not torch.allclose(original.logits[1], donors['R'].logits[1])
    gradient = torch.autograd.grad(original.logits[3, 5], original.messages)[0]
    assert torch.count_nonzero(gradient[:, len(original.prompt) + 3:]) == 0


def test_selection_uses_original_projected_head_energy_and_lexical_ties(model):
    original, _, _ = worlds(model)
    selection, scores = top_heads(model, original, count=4)
    for layer, decoder in enumerate(model.model.layers):
        messages = original.messages[layer][0, original.carrier_positions].float()
        for head in range(original.heads):
            begin = head * original.head_dim
            weight = decoder.self_attn.o_proj.weight[:, begin:begin + original.head_dim].float()
            projected = messages[:, begin:begin + original.head_dim] @ weight.T
            residual = original.residuals[layer][0, original.carrier_positions].float()
            expected = projected.norm(dim=-1) / (residual.norm(dim=-1) + 1e-8)
            torch.testing.assert_close(scores[:, layer, head], expected)
    assert selection.shape == (5, 4, 2)
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.data.zero_()
    tied, _ = top_heads(model, original, count=4)
    assert tied[0].tolist() == [[0, 0], [0, 1], [0, 2], [1, 0]]


def test_noop_patch_and_hook_cleanup_preserve_native_model(model):
    original, _, _ = worlds(model)
    direction = torch.ones_like(original.logits)
    effect = finite_effect(model, original, 0, 1, 1, torch.zeros(8), direction)
    torch.testing.assert_close(effect, torch.zeros_like(effect), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match='intentional'):
        with capture_messages(model), patch_message(model, 0, 1, 4, torch.ones(8)):
            raise RuntimeError('intentional')
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks
    assert not model.model.layers[0].input_layernorm._forward_pre_hooks
    replay = native_trace(model, original.prompt, original.answer, gradients=False)
    torch.testing.assert_close(replay.logits, original.logits, rtol=0, atol=0)


def test_alignment_keeps_unknown_separate_from_zero(model):
    original, donors, alignments = worlds(model)
    selection, _ = top_heads(model, original)
    deltas, available, _ = aligned_deltas(original, donors, alignments, selection)
    assert available.tolist() == [False, True, True, True, True]
    assert torch.isnan(deltas[:, 0]).all()
    assert torch.isfinite(deltas[:, 1:]).all()


def test_variable_length_donors_use_explicit_carrier_alignment(model):
    original = native_trace(model, [1, 2, 3], [4, 5, 6, 7])
    donors = {name: native_trace(model, [1, 2, 3], [first, 12, 5, 6, 7], gradients=False)
              for name, first in zip(('R', 'E', 'U'), (9, 10, 11))}
    maps = {name: [-1, 2, 3, 4] for name in donors}
    selection = torch.tensor([[[0, 1]]] * 4)
    delta, _, _ = aligned_deltas(original, donors, maps, selection)
    expected = donors['R'].messages[0][0, 5, 8:16] - original.messages[0][0, 4, 8:16]
    torch.testing.assert_close(delta[0, 1, 0], expected)
    result = directional_vjp(original, donors, maps, selection)
    assert result['valid'][1, 0, 2]
    assert not result['valid'][2, 0, 1]


def test_dense_vjp_self_aligned_directions_match_small_finite_effects(model):
    original, donors, alignments = worlds(model)
    selection = torch.tensor([[[0, 0], [0, 1]]] * len(original.answer))
    result = directional_vjp(original, donors, alignments, selection)
    assert result['eta'].shape == (3, 3, 5, 2, 5)
    assert result['backward_calls'] == 9
    assert not result['valid'][0].any()  # Edit token has no aligned donor carrier.
    assert not result['valid'][-1].any()  # Terminal carrier cannot affect an earlier prediction.
    assert torch.isnan(result['eta'][:, :, -1]).all()
    deltas, _, _ = aligned_deltas(original, donors, alignments, selection)
    for direction_index, donor in enumerate(donors.values()):
        direction = direction_vector(original.logits.log_softmax(-1), donor.logits.log_softmax(-1))
        assert not direction.requires_grad
        torch.testing.assert_close(direction.sum(-1), torch.zeros(5, dtype=torch.float64), atol=1e-15, rtol=0)
        for donor_index in range(3):
            for slot in range(2):
                finite = finite_effect(model, original, 0, slot, 1, deltas[donor_index, 1, slot],
                                       direction, alpha=.01)
                torch.testing.assert_close(finite[:2], torch.zeros_like(finite[:2]), rtol=0, atol=0)
                predicted = result['eta'][direction_index, donor_index, 1, slot, 2:]
                # Native Llama RMS evaluates its norm in FP32, including in a double model.
                # Compare the complete response vector above that finite-difference floor.
                relative_error = (finite[2:] / .01 - predicted).norm() / predicted.norm()
                assert relative_error < .005
