from types import SimpleNamespace

import numpy as np

from experiments.reanchor_flow.rhythm import (
    build_rhythm,
    pair_peaks,
    rolling_delta,
)


def test_rolling_delta_uses_only_previous_events():
    values = np.array([0.0, 0.0, 0.0, 2.0])
    result = rolling_delta(values, 3)
    assert np.isnan(result[:3]).all()
    assert result[3] == 2.0


def test_revisit_peak_pairs_with_later_anchor():
    layers, events = 2, 12
    trace = SimpleNamespace(
        route_change=np.zeros((layers, events)),
        far_prompt_share=np.zeros((layers, events)),
        evidence_share=np.zeros((layers, events)),
        history_share=np.zeros((layers, events)),
        prompt_breadth=np.full((layers, events), 0.4),
        future_influence=np.zeros((layers, events)),
    )
    trace.route_change[:, 6] = 1.0
    trace.far_prompt_share[:, 6] = 1.0
    trace.evidence_share[:, 6] = 0.8
    trace.future_influence[:, 7] = 1.0
    result = build_rhythm(
        trace,
        revisit_window=3,
        peak_quantile=0.8,
        max_lag=2,
    )
    assert result.revisit_peaks[6]
    assert result.anchor_peaks[7]
    assert result.paired_anchor[6] == 7
    assert result.coupling_rate == 1.0


def test_pair_peaks_does_not_match_past_anchor():
    revisit = np.array([False, False, True, False])
    anchor = np.array([False, True, False, False])
    assert pair_peaks(revisit, anchor, 2)[2] == -1
