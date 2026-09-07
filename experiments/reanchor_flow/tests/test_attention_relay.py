"""Numerical tests of layer order, real HF messages, head identity and drift."""
import numpy as np
import pytest
import torch

from experiments.reanchor_flow.attention_relay import RelayObserver, select_relays
from experiments.reanchor_flow.attention_rhythm import RhythmConfig
from experiments.reanchor_flow.attention_rhythm_report import (
    analyze_rhythm, label_observations, matched_label_gap, prompt_history_trends, summarize_head_pairs,
)
from experiments.reanchor_flow.tests.test_attention_rhythm import collect


def relay_fixture():
    n, p = 16, 4
    a = torch.zeros(2, 2, n, n)
    a[..., 0] = 1
    for q in range(n):
        a[0, 0, q].zero_()
        a[0, 0, q, max(q - 1, 0)] = 1
    a[0, 0, 6].zero_()
    a[0, 0, 6, 0] = 1
    for q in (8, 9):
        a[1, 1, q].zero_()
        a[1, 1, q, 6] = 1
    trace = collect(a, p, RhythmConfig(future_lo=2, future_hi=3))
    trace.update(token_ids=np.arange(n), token_text=np.asarray([str(i) for i in range(n)]),
                 local_heads=np.array([0]), global_heads=np.array([3]))
    return trace


def test_relay_uses_same_carrier_and_strictly_deeper_reader_not_adjacent_token():
    trace = relay_fixture()
    audit = analyze_rhythm(trace)
    paths, scores = select_relays(trace, audit)
    np.testing.assert_array_equal(paths, [[0, 0, 0, 6, 1, 1, 8]])
    assert scores.item() == 1
    assert audit['same_carrier_deeper_observed'][0, 3] == 1
    audit['fai_peaks'][:] = False
    audit['fai_peaks'][0, 1, 2] = True  # same layer cannot read a freshly written state
    assert len(select_relays(trace, audit)[0]) == 0
    audit['fai_peaks'][:] = False
    audit['fai_peaks'][1, 1, 3] = True  # the next token is a different carrier
    assert len(select_relays(trace, audit)[0]) == 0


def test_drift_can_be_explained_entirely_by_growing_history_opportunity():
    trace = relay_fixture()
    p = int(trace['response_start'])
    expected = p / (trace['row_position'] + 1)
    trace['attention_buckets'][:] = 0
    trace['attention_buckets'][..., 0] = expected
    trace['attention_buckets'][..., 3] = 1 - expected
    trends = prompt_history_trends(trace)
    assert np.all(trends['prompt_slope_all'] < 0)
    assert np.all(trends['history_slope_all'] > 0)
    np.testing.assert_allclose(trends['prompt_excess_uniform_slope_all'], 0, atol=1e-7)
    labels = np.zeros(len(expected)-1, int)
    labels[5] = 1
    observed = label_observations(trace, labels)
    assert np.isnan(observed['prompt_slope_fully_nonhallucinated']).all()
    assert np.isfinite(observed['prompt_slope_nonhallucinated']).all()


def test_fai_label_join_uses_carrier_not_predictor():
    trace = relay_fixture()
    labels = np.zeros(len(trace['token_ids']) - int(trace['response_start']), int)
    labels[2] = 1
    trace['fai'][:] = 0
    trace['fai'][..., 3] = 12  # carrier b=P+2, whereas predictor q=b-1 is slot 2
    result = label_observations(trace, labels)
    np.testing.assert_array_equal(result['fai_carrier_matched_h_minus_n'], 12)


def test_missing_future_does_not_erase_other_comparable_position_bins():
    values = np.array([[[0., 2., 8., 0., 0., 0., 0., np.nan, np.nan]]])
    labels = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(matched_label_gap(values, labels, np.arange(9)), [[6.]])


def test_pair_cohort_aligns_exact_heads_and_balances_sources(tmp_path):
    entries = []
    for sample, source, lift in [('a1', 'a', 1.), ('a2', 'a', 3.), ('b', 'b', 6.)]:
        np.savez(tmp_path / f'{sample}.npz', distance_mean=np.zeros((2, 2)))
        np.savez(tmp_path / f'{sample}.audit.npz', same_carrier_write_heads=[0],
                 same_carrier_read_heads=[3], same_carrier_deeper_lift=[[lift]])
        entries.append(dict(source_id=source, path=f'{sample}.npz'))
    result = summarize_head_pairs(entries, tmp_path, tmp_path / 'pairs.npz')
    assert result['mean_lift'][0, 3] == 4  # source means: (2 + 6) / 2
    assert result['valid_sources'][0, 3] == 2
    assert np.isfinite(result['mean_lift']).sum() == 1


@pytest.mark.parametrize('zero_read_value', [False, True])
def test_native_relay_messages_and_signed_modules_match_real_hf_llama(zero_read_value):
    transformers = pytest.importorskip('transformers')
    from experiments.common.llama_message_intervention import forward_layers

    cfg = transformers.LlamaConfig(vocab_size=61, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
    cfg._attn_implementation = 'eager'
    torch.manual_seed(17)
    model = transformers.LlamaForCausalLM(cfg).eval()
    if zero_read_value:
        with torch.no_grad():
            model.model.layers[2].self_attn.v_proj.weight.zero_()
    ids = torch.arange(14)
    paths = np.array([[0, 0, 1, 5, 2, 3, 10]])
    with torch.no_grad():
        hidden = model.get_input_embeddings()(ids[None])
        raw = forward_layers(model, hidden, 0, attention_query_chunk=3, apply_final_norm=False)
        observer = RelayObserver(model, ids, raw, paths)
        forward_layers(model, hidden, 0, observer=observer, attention_query_chunk=2)
        trace = observer.finish()
        native = model(ids[None], output_attentions=True, output_hidden_states=True, use_cache=False)
        for edge, (l, h, s, q) in enumerate(trace['relay_edge_index']):
            layer = model.model.layers[l]
            normalized = layer.input_layernorm(native.hidden_states[l])
            values = layer.self_attn.v_proj(normalized)[0].reshape(len(ids), 2, 8)
            coefficient = native.attentions[l][0, h, q, s]
            expected = layer.self_attn.o_proj.weight[:, h*8:(h+1)*8] @ (coefficient * values[s, h//2])
            np.testing.assert_allclose(trace['relay_edge_message'][edge], expected.numpy(), atol=3e-6)
        nodes = trace['relay_node_position']
        observed, runner = trace['relay_observed_token_ids'], trace['relay_runner_token_ids']
        expected = native.logits[0, nodes, observed] - native.logits[0, nodes, runner]
        np.testing.assert_allclose(trace['relay_final_margin'], expected.numpy(), atol=3e-6)
    assert trace['relay_head_code'].shape == (3, 4, 2, 8)
    if zero_read_value:
        assert trace['relay_edge_attention'][1] > 0
        np.testing.assert_array_equal(trace['relay_edge_message'][1], 0)
    assert np.any(trace['relay_head_margin'] < 0) and np.any(trace['relay_head_margin'] > 0)
    reconstructed = (trace['relay_stage_margin'][0] + trace['relay_head_margin'].sum((0, 1))
                     + trace['relay_mlp_margin'].sum(0) + trace['relay_attention_rounding_margin'].sum(0)
                     + trace['relay_residual_rounding_margin'].sum(0)
                     + trace['relay_logit_rounding_margin'] + trace['relay_readout_bias'])
    np.testing.assert_allclose(reconstructed, trace['relay_final_margin'], atol=1e-6)
    np.testing.assert_allclose(trace['relay_residual'][-1], raw[0, nodes].numpy(), atol=3e-6)
