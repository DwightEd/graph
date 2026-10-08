"""Scientific CPU checks for whole-prefix intervention and real past-KV relay."""
import numpy as np
import torch

from .mechanism import prefill_past
from .relay import (amplify_source, four_conditions, prefill_amplified,
                    query_response, result_arrays, strict_past)
from .test_mechanism import small_model


SOURCE = [False, True, True]
PROMPT = 3
TOKENS = [1, 2, 3, 4, 5, 6, 7]
LAYER = 1


@torch.no_grad()
def full_response(model, position, active_positions=None, amplitude=.25):
    prefix = TOKENS[:position + 1]
    with amplify_source(model, SOURCE, PROMPT, amplitude, LAYER,
                        active_positions=active_positions):
        # A fresh complete prefix independently recomputes every native FFN/KV.
        output = model.model(input_ids=torch.tensor([prefix]), use_cache=True)
        logits = model.lm_head(output.last_hidden_state[0, -1]).float()
    actual = TOKENS[position + 1]
    return torch.stack((logits.log_softmax(-1)[actual], logits[actual] - logits[0]))


def test_native_prefix_query_and_cache_cut_exclude_future_without_mutation():
    model = small_model()
    cache = prefill_past(model, TOKENS[:-1])
    original = [layer.keys.clone() for layer in cache.layers]
    for position in (2, 4, 5):
        response, audit = query_response(model, cache, position, TOKENS[position],
            TOKENS[position + 1], 0, SOURCE, PROMPT, layer=LAYER)
        expected = full_response(model, position, amplitude=0.)
        torch.testing.assert_close(response, expected, rtol=2e-6, atol=2e-7)
        assert strict_past(cache, position).get_seq_length() == position
        assert audit['reconstruction_max_error'] < 1e-12
    assert cache.get_seq_length() == len(TOKENS) - 1
    for layer, saved in zip(cache.layers, original):
        torch.testing.assert_close(layer.keys, saved, rtol=0, atol=0)


def test_zero_dose_sham_and_first_target_have_no_past_relay():
    model = small_model()
    native = prefill_past(model, TOKENS[:-1])
    sham, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, 0., LAYER)
    for position in (2, 4, 5):
        result = four_conditions(model, native, sham, position, TOKENS[position],
            TOKENS[position + 1], 0, SOURCE, PROMPT, amplitude=0., layer=LAYER)
        torch.testing.assert_close(result['regret'], torch.zeros_like(result['regret']), rtol=0, atol=0)
    donor, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER)
    first = four_conditions(model, native, donor, 2, TOKENS[2], TOKENS[3], 0,
                            SOURCE, PROMPT, layer=LAYER)
    torch.testing.assert_close(first['response'][0], first['response'][2], rtol=0, atol=0)
    torch.testing.assert_close(first['response'][1], first['response'][3], rtol=0, atol=0)


def test_donor_changes_all_later_past_kv_but_not_preinjection_layers():
    model = small_model()
    native = prefill_past(model, TOKENS[:-1])
    donor, audit = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER)
    assert audit['modified_queries'] == len(TOKENS) - 1 - (PROMPT - 1)
    for index in range(LAYER + 1):
        torch.testing.assert_close(native.layers[index].keys, donor.layers[index].keys, rtol=0, atol=0)
        torch.testing.assert_close(native.layers[index].values, donor.layers[index].values, rtol=0, atol=0)
    assert not torch.allclose(native.layers[LAYER + 1].keys[:, :, PROMPT - 1:],
                              donor.layers[LAYER + 1].keys[:, :, PROMPT - 1:], rtol=1e-8, atol=1e-12)
    for native_layer, donor_layer in zip(native.layers, donor.layers):
        torch.testing.assert_close(native_layer.keys[:, :, :PROMPT - 1],
                                   donor_layer.keys[:, :, :PROMPT - 1], rtol=0, atol=0)


def test_all_conditions_equal_explicit_complete_prefix_and_chunked_donor():
    model = small_model()
    native = prefill_past(model, TOKENS[:-1], chunk_size=2)
    donor, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER, chunk_size=2)
    whole, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER, chunk_size=6)
    for chunked_layer, whole_layer in zip(donor.layers, whole.layers):
        torch.testing.assert_close(chunked_layer.keys, whole_layer.keys, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(chunked_layer.values, whole_layer.values, rtol=1e-12, atol=1e-12)
    position = 5
    result = four_conditions(model, native, donor, position, TOKENS[position],
        TOKENS[position + 1], 0, SOURCE, PROMPT, layer=LAYER)
    schedules = ([], [position], list(range(PROMPT - 1, position)),
                 list(range(PROMPT - 1, position + 1)))
    for index, active in enumerate(schedules):
        expected = full_response(model, position, active_positions=active)
        torch.testing.assert_close(result['response'][index], expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(result['interaction'], result['regret'][2] -
                               result['regret'][0] - result['regret'][1], rtol=0, atol=0)


def test_current_ffn_recomputes_and_past_effect_enters_later_ffn():
    model = small_model()
    native = prefill_past(model, TOKENS[:-1])
    donor, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER)
    ffn = [[] for _ in model.model.layers]
    handles = [layer.mlp.register_forward_hook(
        lambda module, inputs, output, index=index: ffn[index].append(output.detach().clone()))
        for index, layer in enumerate(model.model.layers)]
    try:
        four_conditions(model, native, donor, 5, TOKENS[5], TOKENS[6], 0,
                        SOURCE, PROMPT, layer=LAYER)
    finally:
        for handle in handles:
            handle.remove()
    native_ffn, current_ffn, past_ffn, both_ffn = ffn[LAYER]
    torch.testing.assert_close(native_ffn, past_ffn, rtol=0, atol=0)
    torch.testing.assert_close(current_ffn, both_ffn, rtol=0, atol=0)
    assert not torch.allclose(native_ffn, current_ffn, rtol=1e-9, atol=1e-12)
    assert not torch.allclose(ffn[LAYER + 1][0], ffn[LAYER + 1][2], rtol=1e-9, atol=1e-12)


def test_archived_D_readouts_and_saved_scalar_regret_use_the_same_fixed_candidates():
    model = small_model()
    native = prefill_past(model, TOKENS[:-1])
    donor, _ = prefill_amplified(model, TOKENS[:-1], SOURCE, PROMPT, .25, LAYER)
    rows = []
    frozen = dict(actual_id=np.array([TOKENS[5], TOKENS[6]]), rival_id=np.array([0, 0]),
                  query_position=np.array([4, 5]), actual_logp=[], margin=[])
    for position in (4, 5):
        baseline, _ = query_response(model, native, position, TOKENS[position],
            TOKENS[position + 1], 0, SOURCE, PROMPT, layer=LAYER)
        result = four_conditions(model, native, donor, position, TOKENS[position],
            TOKENS[position + 1], 0, SOURCE, PROMPT, layer=LAYER, native_response=baseline)
        frozen['actual_logp'].append(float(baseline[0]))
        frozen['margin'].append(float(baseline[1]))
        rows.append({name: result[name].numpy() for name in ('response', 'regret', 'interaction')})
    frozen['actual_logp'] = np.array(frozen['actual_logp'])
    frozen['margin'] = np.array(frozen['margin'])
    arrays = result_arrays(rows, frozen, 2)
    np.testing.assert_array_equal(arrays['actual_logp'], arrays['condition_actual_logp'][:, 0])
    np.testing.assert_array_equal(arrays['margin'], arrays['condition_margin'][:, 0])
    for index, name in enumerate(('current', 'past', 'both')):
        assert arrays['regret_' + name].shape == (2,)
        np.testing.assert_array_equal(arrays['regret_' + name],
            arrays['condition_actual_logp'][:, 0] - arrays['condition_actual_logp'][:, index + 1])
    np.testing.assert_array_equal(arrays['interaction'], arrays['regret_both'] -
                                  arrays['regret_current'] - arrays['regret_past'])
