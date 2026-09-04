import numpy as np

from experiments.reanchor_flow.events import match_events, match_onsets


def test_onset_matching_prefers_same_token_nearby():
    label = np.zeros(30, dtype=bool)
    label[18:20] = True
    token = np.arange(30)
    token[8] = token[18]
    pairs = match_onsets(label, token, window=3, caliper=12)
    assert pairs == [(18, 8)]


def test_matching_uses_boundary_then_preoutcome_covariates():
    label = np.zeros(40, dtype=bool)
    label[20] = True
    token = np.arange(40)
    boundary = np.zeros(40, dtype=bool)
    boundary[[10, 20]] = True
    entropy = np.zeros(40)
    entropy[10] = 2.0
    entropy[19] = 2.0
    pairs = match_onsets(
        label,
        token,
        window=3,
        caliper=12,
        covariates={"entropy": entropy},
        boundary=boundary,
    )
    assert pairs == [(20, 10)]


def test_matching_rejects_misaligned_covariates():
    label = np.zeros(20, dtype=bool)
    token = np.arange(20)
    try:
        match_onsets(label, token, covariates={"entropy": np.zeros(19)})
    except ValueError as error:
        assert "entropy" in str(error)
    else:
        raise AssertionError("misaligned covariate was accepted")


def test_internal_event_matching_excludes_nearby_events():
    event = np.zeros(40, dtype=bool)
    event[20] = True
    token = np.arange(40)
    token[18] = token[20]
    token[10] = token[20]
    pairs = match_events(event, token, window=2, caliper=12)
    assert pairs == [(20, 10)]
