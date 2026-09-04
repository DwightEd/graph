import numpy as np

from experiments.reanchor_flow.report import coupling_population, interval_direction


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
