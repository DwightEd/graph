"""Native tiny-Llama capture contracts; CPU only, no QA effect claim."""
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from state_audit.model.adapter import ModelAdapter

from .capture import (allowed_keys, capture_answer, local_key_positions,
                      message_groups)


@pytest.fixture
def adapter():
    torch.manual_seed(73)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.)
    return ModelAdapter(LlamaForCausalLM(config).float())


def case():
    return [1, 2, 3, 4, 5], [6, 7, 8, 9], [False, True, True, False, False]


def test_source_mask_preserves_source_queries_but_blocks_other_receivers():
    source = torch.tensor([False, True, True, False, False])
    positions = torch.arange(5)
    blocked = allowed_keys(source, positions, True)
    assert blocked[2, 1] and blocked[2, 2]
    assert not blocked[3, 1] and not blocked[3, 2]
    assert blocked[3, 0] and blocked[3, 3]
    assert not blocked[1, 2]
    torch.testing.assert_close(allowed_keys(source, positions, False),
                               torch.ones(5, 5, dtype=torch.bool).tril())


def test_local_neighbors_are_strict_past_answer_nodes_and_do_not_include_self():
    positions = torch.tensor([4, 5, 6, 7])
    neighbors, valid = local_key_positions(positions, 5, 2)
    torch.testing.assert_close(neighbors, torch.tensor([[3, 2], [4, 3], [5, 4], [6, 5]]))
    torch.testing.assert_close(valid, torch.tensor([[False, False], [False, False],
                                                  [True, False], [True, True]]))
    assert (neighbors < positions[:, None]).all()


def test_group_sums_preserve_native_values_and_distinct_receiver_states():
    attention = torch.tensor([[[.2, .3, .5, 0., 0.]], [[.1, .2, .3, .1, .3]]])
    values = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
    source = torch.tensor([False, True, False, False, False])
    positions = torch.tensor([2, 4])
    messages, masses = message_groups(attention, values, source, positions, 3, 1)
    expected = torch.einsum('rhs,shd->rhd', attention, values)
    torch.testing.assert_close(messages.sum(1), expected)
    torch.testing.assert_close(masses.sum(1), torch.ones(2, 1))
    torch.testing.assert_close(messages[1, 1, 0], attention[1, 0, 3] * values[3, 0])
    assert not torch.equal(expected[0], expected[1])


def test_pair_preserves_ids_full_nodes_head_coordinates_and_both_alignments(adapter):
    prompt, answer, mask = case()
    arrays, audit = capture_answer(adapter, prompt, answer, mask, layer=1,
                                   local_width=2, query_chunk=2)
    assert arrays['nodes'].shape == (2, 5, 3, 16)
    assert arrays['query'].shape == (2, 5, 4, 4)
    assert arrays['key'].shape == arrays['value'].shape == (2, 9, 2, 4)
    assert arrays['local_attention'].shape == (2, 5, 4, 2)
    assert arrays['group_heads'].shape == (2, 5, 5, 4, 4)
    np.testing.assert_array_equal(arrays['token_ids'], prompt + answer)
    np.testing.assert_array_equal(arrays['query_positions'][:-1], [4, 5, 6, 7])
    np.testing.assert_array_equal(arrays['query_positions'][1:], [5, 6, 7, 8])
    assert arrays['actual_logp'].shape == arrays['entropy'].shape == (2, 4)
    assert np.all(arrays['local_attention'][:, ~arrays['local_valid'].any(1)] == 0)
    for world in audit['worlds']:
        assert world['state_equation_max_abs'] < 1e-7
        assert world['head_reconstruction_max_abs'] < 1e-7
        assert world['write_reconstruction_max_abs'] < 1e-7
        assert world['group_sum_max_abs'] < 1e-7
        assert world['forbidden_attention_max_abs'] == 0
    assert audit['worlds'][1]['blocked']
    assert np.count_nonzero(arrays['nodes'][0] != arrays['nodes'][1]) > 0


def test_blocked_source_message_is_zero_and_other_groups_remain(adapter):
    prompt, answer, mask = case()
    arrays, _ = capture_answer(adapter, prompt, answer, mask, layer=1,
                               local_width=2, stop_after_layer=True)
    assert np.count_nonzero(arrays['group_heads'][1, :, 0]) == 0
    assert np.count_nonzero(arrays['group_mass'][1, :, 0]) == 0
    assert np.count_nonzero(arrays['group_heads'][1, :, 3]) > 0


def test_expected_early_exit_skips_higher_layers_preserves_states_and_removes_hooks(adapter):
    prompt, answer, mask = case()
    full, full_audit = capture_answer(adapter, prompt, answer, mask, layer=1, local_width=2)
    visits = []
    handle = adapter.layers[2].register_forward_hook(lambda module, args, output: visits.append(1))
    early, early_audit = capture_answer(adapter, prompt, answer, mask, layer=1,
                                       local_width=2, stop_after_layer=True)
    handle.remove()
    assert visits == []
    assert full_audit['full_model_forwards'] == 2
    assert early_audit['full_model_forwards'] == 0
    assert [world['executed_layers'] for world in early_audit['worlds']] == [2, 2]
    assert all(not world['full_model_completed'] for world in early_audit['worlds'])
    assert 'actual_logp' not in early and 'entropy' not in early
    for name, values in early.items():
        np.testing.assert_array_equal(values, full[name])
    for module in adapter.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_unexpected_forward_failure_is_not_swallowed_and_hooks_are_removed(adapter):
    prompt, answer, mask = case()
    def fail(module, args, output):
        raise RuntimeError('native failure')
    handle = adapter.layers[0].register_forward_hook(fail)
    with pytest.raises(RuntimeError, match='native failure'):
        capture_answer(adapter, prompt, answer, mask, layer=1, stop_after_layer=True)
    handle.remove()
    for module in adapter.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize('paired', [False, True])
def test_future_answer_rows_do_not_change_prefix_nodes_or_messages(adapter, paired):
    prompt, answer, mask = case()
    bulk, _ = capture_answer(adapter, prompt, answer, mask, layer=1,
                             local_width=2, paired=paired, stop_after_layer=True)
    prefix, _ = capture_answer(adapter, prompt, answer[:2], mask, layer=1,
                               local_width=2, paired=paired, stop_after_layer=True)
    for name in ('nodes', 'query', 'local_attention', 'group_heads', 'group_mass'):
        np.testing.assert_allclose(bulk[name][:, :3], prefix[name], rtol=1e-6, atol=1e-7)
    for name in ('key', 'value'):
        np.testing.assert_allclose(bulk[name][:, :7], prefix[name], rtol=1e-6, atol=1e-7)


def test_empty_source_pair_is_identity_without_dropping_positions(adapter):
    prompt, answer, _ = case()
    arrays, _ = capture_answer(adapter, prompt, answer, [False] * len(prompt), layer=1,
                               local_width=2, stop_after_layer=True)
    for name in ('nodes', 'query', 'key', 'value', 'local_attention', 'group_heads', 'group_mass'):
        np.testing.assert_array_equal(arrays[name][0], arrays[name][1])


def test_full_forward_probability_is_next_token_aligned_to_actual_observer(adapter):
    prompt, answer, mask = case()
    arrays, _ = capture_answer(adapter, prompt, answer, mask, layer=1,
                               local_width=2, paired=False)
    with torch.no_grad():
        logits = adapter.native(adapter.input_ids(prompt + answer), use_cache=False).logits[0]
        logp = logits[len(prompt) - 1:-1].log_softmax(-1)
        expected = logp.gather(1, torch.tensor(answer)[:, None])[:, 0].numpy()
    np.testing.assert_allclose(arrays['actual_logp'][0], expected, rtol=1e-6, atol=1e-7)
