import numpy as np
import pytest

from experiments.reanchor_flow.message_dag.hurdle import summarize_hurdle


def test_hurdle_separates_event_incidence_from_conditional_transport():
    samples = [
        {
            "group": "test/QA",
            "source": "s1",
            "labels": np.array([0, 0, 1, 1]),
            "ordinary": np.ones(4, dtype=bool),
            "event": np.array([False, True, False, True]),
            "transport": np.array([np.nan, 0.2, np.nan, 0.7]),
        },
        {
            "group": "test/QA",
            "source": "s2",
            "labels": np.array([0, 0, 1, 1]),
            "ordinary": np.ones(4, dtype=bool),
            "event": np.array([False, False, True, True]),
            "transport": np.array([np.nan, np.nan, 0.4, 0.6]),
        },
    ]

    report = summarize_hurdle(samples, bootstrap=0)["test/QA"]

    assert report["N"]["event_rate"]["mean"] == 0.25
    assert report["H"]["event_rate"]["mean"] == 0.75
    assert report["H_minus_N"]["event_rate"]["mean"] == 0.5
    assert report["N"]["transport_given_event"]["mean"] == 0.2
    assert report["H"]["transport_given_event"]["mean"] == 0.6
    assert report["H_minus_N"]["transport_given_event"]["mean"] == pytest.approx(0.4)
    assert report["N"]["tokens"] == 4
    assert report["N"]["events"] == 1
    assert report["H"]["tokens"] == 4
    assert report["H"]["events"] == 3
    assert report["estimand"] == "event incidence plus transport conditional on a prior event"


def test_hurdle_reports_missing_transport_without_zero_imputation():
    samples = [{
        "group": "test/QA",
        "source": "s1",
        "labels": np.array([0, 1, 1]),
        "ordinary": np.ones(3, dtype=bool),
        "event": np.array([False, True, True]),
        "transport": np.array([np.nan, np.nan, 0.5]),
    }]

    report = summarize_hurdle(samples, bootstrap=0)["test/QA"]

    assert report["H"]["events"] == 2
    assert report["H"]["transport_scored"] == 1
    assert report["H"]["transport_coverage_given_event"] == 0.5
    assert report["H"]["transport_given_event"]["mean"] == 0.5
    assert np.isnan(report["N"]["transport_given_event"]["mean"])
