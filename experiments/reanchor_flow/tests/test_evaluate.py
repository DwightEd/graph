import numpy as np

from experiments.reanchor_flow.events import match_onsets


def test_onset_matching_prefers_same_token_nearby():
    label = np.zeros(30, dtype=bool)
    label[18:20] = True
    token = np.arange(30)
    token[8] = token[18]
    pairs = match_onsets(label, token, window=3, caliper=12)
    assert pairs == [(18, 8)]
