"""Causal shift, AV partition, native suffix and directional-control checks."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .mechanism import (channel_masks, finite_query, finite_query_group, measure_query,
                        norm_preserving_control, prefill_past, signed_responses)


def small_model():
    torch.manual_seed(18)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=3, num_attention_heads=3,
                        num_key_value_heads=1, attention_dropout=0.,
                        _attn_implementation='sdpa')
    return LlamaForCausalLM(config).double().eval().requires_grad_(False)


def test_masks_partition_only_current_and_past_keys():
    masks = channel_masks([False, True, True], 3, 8, 2, 'cpu')
    assert masks.sum(0).tolist() == [1] * 9
    assert torch.where(masks[1])[0].tolist() == [6, 7]
    assert torch.where(masks[2])[0].tolist() == [3, 4, 5]
    assert torch.where(masks[3])[0].tolist() == [8]


def test_native_cached_identity_av_partition_and_suffix_derivative():
    model = small_model()
    past = prefill_past(model, [1, 2, 3, 4], chunk_size=2)
    measured = measure_query(model, past, 5, 6, [False, True, True], 3, local_window=1)
    assert past.get_seq_length() == 4
    expected = model(torch.tensor([[1, 2, 3, 4, 5]])).logits[0, -1].float().log_softmax(-1)
    torch.testing.assert_close(measured['logp_full'], expected, rtol=2e-5, atol=2e-6)
    assert measured['reconstruction_error'].max() < 2e-7
    torch.testing.assert_close(measured['response'], measured['response_residual'] + measured['response_ffn'])
    layer, group, head = 0, 0, 1
    direction = measured['messages'][layer, group, head]
    dose = .01
    positive = finite_query(model, past, 5, 6, measured['rival_id'], layer, head, direction, dose)
    negative = finite_query(model, past, 5, 6, measured['rival_id'], layer, head, direction, -dose)
    slope = (positive - negative) / (2 * dose)
    torch.testing.assert_close(slope.float(), measured['response'][:, layer, group, head].float(),
                               rtol=.015, atol=2e-6)


def test_direction_control_keeps_each_projected_norm_and_changes_response():
    model = small_model()
    past = prefill_past(model, [1, 2, 3, 4])
    measured = measure_query(model, past, 5, 6, [False, True, True], 3)
    message, gram = measured['messages'], measured['output_gram']
    random_message = norm_preserving_control(message, gram)
    original_norm = torch.einsum('lghd,lhde,lghe->lgh', message, gram, message)
    random_norm = torch.einsum('lghd,lhde,lghe->lgh', random_message, gram, random_message)
    torch.testing.assert_close(original_norm, random_norm, rtol=1e-5, atol=1e-8)
    assert not torch.allclose(measured['response'], signed_responses(measured['gradient'], random_message))


def test_zero_ffn_removes_only_within_layer_ffn_mediated_response():
    model = small_model()
    for layer in model.model.layers:
        layer.mlp.down_proj.weight.data.zero_()
    past = prefill_past(model, [1, 2, 3, 4])
    measured = measure_query(model, past, 5, 6, [False, True, True], 3)
    torch.testing.assert_close(measured['response_ffn'], torch.zeros_like(measured['response_ffn']),
                               rtol=0, atol=1e-8)


def test_site_sum_is_joint_frozen_direction_derivative():
    model = small_model()
    past = prefill_past(model, [1, 2, 3, 4])
    measured = measure_query(model, past, 5, 6, [False, True, True], 3)
    directions = measured['messages'][:, 0]
    arguments = (model, past, 5, 6, measured['rival_id'], directions)
    dose = .01
    positive = finite_query_group(*arguments, dose=dose)
    negative = finite_query_group(*arguments, dose=-dose)
    predicted = measured['response'][:, :, 0].sum((1, 2))
    torch.testing.assert_close((positive - negative) / (2 * dose), predicted, rtol=.02, atol=3e-6)
    for layer in model.model.layers:
        assert not layer.self_attn.o_proj._forward_pre_hooks
