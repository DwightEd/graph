import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .capture import distribution_effect, receiver_mask, source_mask
from .model import joint_profile, js, regimes, source_spans
from .state_path import path_spans, source_path


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
