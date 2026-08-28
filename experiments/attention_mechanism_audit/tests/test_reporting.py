from experiments.attention_mechanism_audit.reporting import (
    KEY_METRICS,
    ONSET_METRICS,
    render_report,
)


def _report():
    keys = {metric.key for metric in KEY_METRICS}
    keys.update(metric.key for metric in ONSET_METRICS)
    summary = {
        "correct_mean": 0.2,
        "hallucinated_mean": 0.3,
        "position_matched_source_equal_difference": 0.04,
        "ci95": [0.01, 0.07],
        "sources": 12,
        "matched_cells": 30,
    }
    onset = {
        "offset": [-1, 0, 1],
        "difference_in_difference": [0.0, 0.03, 0.04],
        "ci95_low": [0.0, 0.01, 0.01],
        "ci95_high": [0.0, 0.05, 0.07],
        "events": [8, 8, 8],
        "sources": [5, 5, 5],
    }
    return {
        "samples": 20,
        "tokens": 100,
        "hallucinated_tokens": 10,
        "summaries": {key: dict(summary) for key in keys},
        "matched_onset": {key: dict(onset) for key in keys},
    }


def test_key_report_separates_audit_from_detection_and_prints_support_counts():
    text = render_report(_report())

    assert "20 responses | 100 tokens | 10 hallucinated (10.00%)" in text
    assert "Matched analysis: 12 sources, 30 source-position cells" in text
    assert "Routing imbalance (%)" in text
    assert "First hallucinated token" in text
    assert "+3.0000 pp" in text
    assert "detector score and threshold not evaluated" in text
    assert "Definitions" not in text
    assert "All saved metrics" not in text


def test_explanation_is_opt_in():
    text = render_report(_report(), explain=True)

    assert "Definitions" in text
    assert "edge mass = attention" in text
    assert "Limits" in text


def test_all_metrics_are_opt_in():
    report = _report()
    report["summaries"]["diagnostic_only"] = {
        "correct_mean": None,
        "hallucinated_mean": None,
        "position_matched_source_equal_difference": None,
        "ci95": [None, None],
        "sources": 0,
        "matched_cells": 0,
    }

    text = render_report(report, all_metrics=True)

    assert "All saved metrics" in text
    assert "diagnostic_only" in text
    assert "C=n/a H=n/a" in text
