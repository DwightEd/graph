import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .capture import distribution_effect, receiver_mask, source_mask, edited_tokens, forward, distributions, measure_intervention, matching_edits
from .model import joint_profile, js, regimes, source_spans, relation_edits, repeated_source_sets, relation_scores
from .state_path import path_spans, source_path
from .score import score_all_relations
from experiments.decision_risk_flow.data import write_json


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, tokens):
        return ''.join(self.pieces[token] for token in tokens)


def test_source_partition_preserves_negation_decimal_and_all_keys():
    pieces = ['question', " WiFi", ':', " no,", ' stars', ': ', '4', '.', '0', '.', ' Next', ' instruction']
    groups = source_spans(list(range(len(pieces))), [0]+[1]*10+[0], Tokenizer(pieces))
    assert [key for group in groups for key in group['keys']] == list(range(1, 11))
    assert groups[0]['text'] == ' WiFi: no,'
    assert any('4.0' in group['text'] for group in groups)


def test_distribution_js_is_full_vocabulary_and_matches_numpy():
    left = np.array([[.1, .2, .7], [.4, .3, .3]])
    right = np.array([[.1, .7, .2], [.4, .3, .3]])
    actual, measured = distribution_effect(torch.tensor(np.log(left)), torch.tensor(np.log(right)), torch.tensor([0, 1]))
    np.testing.assert_allclose(measured, js(left, right), atol=1e-12)
    assert measured[0] > 0  # Actual token is unchanged; binary JS would miss this.
    np.testing.assert_allclose(actual, np.log([.1, .3]))


def test_joint_ranking_and_variable_regimes_do_not_change_scores():
    roots = np.array([[9., 1.], [9., 1.], [1., 9.], [1., 9.]])
    profile = joint_profile(roots, roots)
    np.testing.assert_allclose(profile.sum(-1), 1)
    spans, change, thresholds = regimes(profile, np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.]))
    assert spans == [dict(start=0, stop=2), dict(start=2, stop=4)]
    assert change[2, 0] > thresholds[0]


def test_native_masks_preserve_prefix_and_sham():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        baseline = model(tokens).logits
        sham = model(tokens, attention_mask=receiver_mask(tokens, 4, [])).logits
        direct = model(tokens, attention_mask=receiver_mask(tokens, 4, [1, 2])).logits
        complete = model(tokens, attention_mask=source_mask(tokens, [1, 2])).logits
        prefix = model(tokens[:, :5], attention_mask=source_mask(tokens[:, :5], [1, 2])).logits
    torch.testing.assert_close(baseline, sham)
    torch.testing.assert_close(direct[:, :4], baseline[:, :4])
    torch.testing.assert_close(complete[:, :5], prefix)
    assert not torch.allclose(direct[:, 4], baseline[:, 4])


def test_source_state_path_refresh_and_future_scope():
    profile = np.array([[.999, .001], [.49, .51], [.999, .001], [.001, .999]])
    stable, _ = source_path(profile, np.zeros(4), np.zeros(4))
    refreshed, _ = source_path(profile, np.ones(4), np.ones(4))
    np.testing.assert_array_equal(refreshed, profile.argmax(-1))
    assert len(path_spans(stable)) < len(path_spans(refreshed))
    assert [t for span in path_spans(refreshed) for t in range(span['start'], span['stop'])] == list(range(4))


def test_source_only_relation_candidates_and_abstention():
    negative = list(relation_edits('Private income is not taxed.'))
    assert len(negative) == 1
    assert negative[0]['flip'] == 'is'
    assert negative[0]['equivalent'] == "isn't"
    boolean = list(relation_edits("'WiFi': 'no', 'OutdoorSeating': True"))
    assert [item['flip'] for item in boolean] == ['yes', 'False']
    assert all(item['control'] == 'ordinary_case' for item in boolean)
    assert [item['equivalent'] for item in boolean] == ['No', 'true']
    bounded = list(relation_edits('For more than four days,'))
    assert bounded[0]['flip'] == 'less than'
    assert bounded[0]['equivalent'] == 'over'
    assert bounded[1]['flip'] == 'exactly'
    assert not list(relation_edits('Cook for 4-5 minutes, until done.'))


def test_repeat_group_does_not_merge_boolean_attributes():
    groups = [dict(index=0, text="'Monday': '9:0-22:30',"),
              dict(index=1, text="'Tuesday': '9:0-22:30',"),
              dict(index=2, text="'Wednesday': '8:0-22:30',"),
              dict(index=3, text="'Outdoor': False, 'Parking': False")]
    assert repeated_source_sets(groups) == [dict(value='9:0-22:30', sources=[0, 1])]


def test_relation_score_removes_control_and_uses_measured_messages():
    original = np.log(np.array([.2, .8]))
    flipped = np.log(np.array([[.6], [.4]]))
    equivalent = original[:, None].copy()
    messages = np.ones((1, 2, 3, 2)) * .5
    gain, gated, _, _, specificity = relation_scores(original, flipped, equivalent, messages, messages*.2)
    assert gain[0, 0] > 0 and gain[1, 0] < 0
    np.testing.assert_allclose(specificity, .8)
    np.testing.assert_allclose(gated, gain*.8)
    _, equal_message_gain, _, _, _ = relation_scores(original, flipped, equivalent, messages, messages)
    np.testing.assert_array_equal(equal_message_gain, 0)


def test_cache_reuse_requires_same_address_and_replacement_tokens():
    previous = [dict(source=1, keys=[3, 8], flip_ids=[1, 2], equivalent_ids=[3, 4])]
    changed_control = dict(source=1, keys=[3, 8], flip_ids=[1, 2], equivalent_ids=[3, 5])
    changed_address = dict(source=2, keys=[8, 12], flip_ids=[1, 2], equivalent_ids=[3, 4])
    assert matching_edits([changed_control, changed_address], previous, 'flip') == [0, -1]
    assert matching_edits([changed_control], previous, 'equivalent') == [-1]


def test_all_candidate_readout_can_change_source_without_overwriting_frozen_scores(tmp_path):
    directory = tmp_path/'sample'
    directory.mkdir()
    write_json(directory/'candidates.json', dict(edits=[dict(source=0), dict(source=1)]))
    np.savez(directory/'scores.npz', selected=np.array([0, 0]))
    unchanged = (directory/'scores.npz').read_bytes()
    original = np.log([.2, .8])
    np.savez(directory/'relation_effects.npz', original_logp=original, token_ids=[5, 6],
        flip_logp=np.log([[.21, .6], [.79, .7]]), equivalent_logp=np.repeat(original[:, None], 2, axis=1),
        flip_message_change=np.ones((2, 2, 3, 2))*.5, equivalent_message_change=np.ones((2, 2, 3, 2))*.1)
    score_all_relations(dict(key='sample', response=dict(answer_ids=[5, 6])), tmp_path)
    assert (directory/'scores.npz').read_bytes() == unchanged
    with np.load(directory/'all_scores.npz') as saved:
        assert saved['selected'].tolist() == [1, 0]
        np.testing.assert_allclose(saved['all_relation_logp'][0], np.log(3))


def test_reused_native_measurement_and_edit_answer_alignment():
    torch.manual_seed(7)
    config = LlamaConfig(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    row = dict(prompt=[1, 2, 3, 4], response=dict(answer_ids=[5, 6, 7]))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    targets = torch.tensor([5, 6, 7])
    hidden, messages = forward(model, tokens, 4, torch.ones_like(tokens))
    baseline = dict(hidden=hidden, messages=messages, distribution=distributions(model, hidden))
    same = measure_intervention(model, tokens, 4, targets, baseline, torch.ones_like(tokens))
    np.testing.assert_allclose(same['js'], 0, atol=1e-7)
    np.testing.assert_allclose(same['message_change'], 0, atol=1e-7)
    candidate = dict(keys=[1, 3], flip_ids=[9])
    changed, length = edited_tokens(row, candidate, 'flip', 'cpu')
    assert length == 3
    assert changed.tolist() == [[1, 9, 4, 5, 6]]
    measured = measure_intervention(model, changed, length, targets, baseline, torch.ones_like(changed))
    with torch.no_grad():
        for target in range(3):
            logits = model(changed[:, :length+target]).logits[0, -1].log_softmax(-1)
            np.testing.assert_allclose(measured['logp'][target], float(logits[targets[target]]), atol=1e-6)
