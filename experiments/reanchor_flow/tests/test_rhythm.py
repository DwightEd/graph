from types import SimpleNamespace

import numpy as np

from experiments.reanchor_flow.rhythm import build_rhythm, pair_peaks, rolling_delta


def test_rolling_delta_uses_only_previous_events():
    values = np.array([0.0, 0.0, 0.0, 2.0])
    result = rolling_delta(values, 3)
    assert np.isnan(result[:3]).all()
    assert result[3] == 2.0


def blank_trace(layers=2, events=12):
    zero = np.zeros((layers, events))
    return SimpleNamespace(
        route_change=zero.copy(),
        prompt_share=zero.copy(),
        evidence_share=zero.copy(),
        history_share=zero.copy(),
        prompt_lift=zero.copy(),
        evidence_lift=zero.copy(),
        history_lift=zero.copy(),
        nonlocality=zero.copy(),
        prompt_breadth=np.full((layers, events), 0.4),
        predictor_reuse=zero.copy(),
        future_influence=zero.copy(),
    )


def test_internal_transition_and_prompt_revisit_are_independent():
    trace = blank_trace()
    trace.route_change[:, 6] = 1.0
    trace.prompt_share[:, 6] = 1.0
    trace.evidence_share[:, 6] = 0.8
    trace.future_influence[:, 7] = 1.0
    result = build_rhythm(trace, revisit_window=3, peak_quantile=0.8, max_lag=2)
    assert result.transition_peaks[6]
    assert result.prompt_peaks[6]
    assert result.anchor_peaks[7]
    assert result.prompt_paired_anchor[6] == 7


def test_nonlocal_review_is_detected_without_prompt_revisit():
    trace = blank_trace()
    trace.nonlocality[:, 5] = 1.0
    trace.future_influence[:, 6] = 1.0
    result = build_rhythm(trace, revisit_window=3, peak_quantile=0.8, max_lag=2)
    assert result.review_peaks[5]
    assert not result.prompt_peaks.any()
    assert result.review_paired_anchor[5] == 6


def test_pair_peaks_does_not_match_past_anchor():
    event = np.array([False, False, True, False])
    anchor = np.array([False, True, False, False])
    assert pair_peaks(event, anchor, 2)[2] == -1
