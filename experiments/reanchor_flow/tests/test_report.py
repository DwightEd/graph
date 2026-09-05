import numpy as np

from experiments.reanchor_flow.report import (
    coupling_population,
    functional_summary,
    interval_direction,
)


def row(source, rate, null, peaks=2):
    peak = np.zeros(6, dtype=bool)
    peak[:peaks] = True
    paired = np.full(6, -1, dtype=np.int64)
    paired[: int(round(rate * peaks))] = np.arange(int(round(rate * peaks)))
    return {
        "source_id": source,
        "prompt_peak": peak,
        "prompt_paired_anchor": paired,
        "prompt_coupling_rate": rate,
        "prompt_coupling_null_rate": null,
        "prompt_median_anchor_lag": 1.0,
    }


def test_population_summary_reports_source_level_lift():
    rows = [
        row("a", 1.0, 0.25),
        row("b", 0.0, 0.25),
    ]
    result = coupling_population(rows, "prompt", bootstrap=0, seed=7)
    assert result["event_peaks"] == 4
    assert result["sample_lift"]["sources"] == 2
    assert result["sample_lift"]["mean"] == 0.25
    assert result["positive_source_fraction"] == 0.5


def test_interval_direction_uses_ci_not_point_estimate():
    positive = {"mean": -99.0, "ci95": [0.1, 0.4]}
    crossing = {"mean": 2.0, "ci95": [-0.1, 0.4]}
    assert interval_direction(positive, "positive") == "supported"
    assert interval_direction(positive, "negative") == "contradicted"
    assert interval_direction(crossing, "positive") == "inconclusive"


def test_functional_summary_uses_every_functional_sample():
    count = 20
    label = np.zeros(count, dtype=bool)
    label[12] = True
    token = np.arange(count)
    token[5] = token[12]
    signal = np.zeros(count)
    signal[12] = 2.0
    row = {
        "source_id": "a",
        "functional": True,
        "label": label,
        "target_token_id": token,
        "prediction_position": np.arange(count),
        "sentence_boundary_position": np.array([], dtype=int),
        "baseline_entropy": np.zeros(count),
        "baseline_target_logprob": np.zeros(count),
        "evidence_share_layer": np.stack((signal, signal)),
        "evidence_effect": signal,
        "context_distribution_js": signal,
        "context_target_logprob_gain": signal,
        "context_adoption_margin": signal,
        "context_target_log_rank": signal,
    }
    result = functional_summary([row], bootstrap=0, seed=7)
    assert result["samples"] == 1
    assert result["onset_pairs"] == 1
    assert result["onset_minus_clean"]["context_adoption_margin"]["mean"] == 2.0
