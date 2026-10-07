"""Scientific contracts for repetition controls and native channel intervention."""
import numpy as np
import torch

from .dominance import channel_messages, intervention_worlds
from .dominance_data import encode_passage, transfer_source_mask
from .repetition_analysis import match_tokens
from .repetition_edges import edge_features
from .repetition_teacher import different_word_persistence, literal_repetition
from .test_grounded_projection import fixture, finite_patch
from .grounded_projection import observed_attention, replay_queries


def test_literal_counts_exclude_current_and_future_occurrences():
    tokens = np.array([1, 2, 1, 2, 1, 2])
    counts, lag = literal_repetition(tokens)
    assert counts[:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    assert counts[5].tolist() == [2, 2, 1]
    assert lag.tolist() == [0, 0, 2, 2, 2, 2]
    assert np.array_equal(counts[:4], literal_repetition(tokens[:4])[0])


def test_explicit_passage_mask_preserves_text_and_matches_saved_prefix():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return 'CHAT['+messages[0]['content']+']'

        def __call__(self, text, add_special_tokens, return_offsets_mapping):
            return dict(input_ids=[ord(c) for c in text],
                        offset_mapping=[(i, i+1) for i in range(len(text))])

    prompt = 'passage 1:ONE\n\npassage 2:TWO\n\npassage 3:THREE\nIn case the passages fail'
    source = dict(source_id='1', prompt=prompt)
    canonical = encode_passage(source, CharacterTokenizer(), 3)
    saved = [999]+canonical['prompt_with_source']+[998]
    mask = transfer_source_mask(canonical, saved)
    selected = ''.join(chr(token) for token, keep in zip(saved, mask) if keep)
    assert selected == 'passage 3:THREE'
    assert not mask[0] and not mask[-1]


def test_pattern_similarity_excludes_same_word_and_future():
    vectors = np.array([[1., 0], [1., 0], [0., 1], [1., 0]])
    tokens = np.array([1, 1, 2, 3])
    scores = different_word_persistence(vectors, tokens)['recent_pattern']
    assert np.isnan(scores[:2]).all()
    assert scores[2] == 0 and scores[3] == 1
    assert np.isnan(different_word_persistence(vectors[:2], tokens[:2])['recent_pattern']).all()


def test_matches_keep_unique_controls_and_prior_error_state():
    row = dict(token_ids=np.ones(10, dtype=int), labels=np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0]))
    counts = literal_repetition(row['token_ids'])[0]
    pairs = match_tokens(row, counts)
    assert pairs == [(3, 2), (6, 5), (8, 7)]
    assert len({c for t, c in pairs}) == len(pairs)
    prior = np.r_[False, row['labels'][:-1].cumsum() > 0]
    assert all(prior[t] == prior[c] for t, c in pairs)


def test_onset_expectation_includes_zero_edges_and_singleton_has_no_excess():
    tokens = np.array([9, 8, 1, 2, 3, 4, 5])
    attention = np.array([[.2, .1, .1, .2, .0, .4, 0.]])
    features = edge_features(attention, tokens, 2, 4, 3, np.array([False, True]))
    assert features['onset'][0] == .4
    assert features['onset_excess'][0] == 0
    assert not features['exchangeable'][0]
    features = edge_features(attention, tokens, 2, 4, 1, np.array([False, True]))
    assert np.isclose(features['onset_excess'][0], .2-(.1+.2)/2)
    assert features['source'][0] == .1


def test_channels_preserve_query_self_and_sham_per_head_norm():
    adapter, tokens, records, hidden, cosine, sine = fixture()
    positions = torch.tensor([3, 5])
    attention, values = observed_attention(adapter, 1, records[1], torch.arange(3, 8), cosine, sine)
    attention = attention[positions-3]
    messages, masses = channel_messages(attention, values, 4, positions,
        [False, True, True, False], [1, 2])
    assert torch.equal(messages['history'][0], torch.zeros(4, 8))
    assert torch.allclose(messages['history'][1]-messages['remote'][1],
        attention[1, :, 5, None]*values[5], atol=1e-7)
    assert torch.allclose(messages['both'], messages['source']+messages['history'])
    names, delta = intervention_worlds(messages, 9)
    for name, message in messages.items():
        assert torch.allclose(delta[names.index('random_'+name)].norm(dim=-1), message.norm(dim=-1), atol=1e-7)
    replay = replay_queries(adapter, records, 1, positions-3, positions,
        delta[names.index('drop_source_1')], cosine, sine)
    for index, position in enumerate(positions):
        with finite_patch(adapter.native, 1, position, -messages['source'][index]):
            native = adapter.native.model(tokens).last_hidden_state[0, position]
        assert torch.allclose(replay[index], native, atol=3e-6)
