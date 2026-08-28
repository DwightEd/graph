from experiments.attention_mechanism_audit.reporting import GROUPS, ONSET, render_report


def _report():
    keys = {metric.key for _title, metrics in GROUPS for metric in metrics}
    keys.update(metric.key for metric in ONSET)
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

    assert "samples=20 tokens=100 hallucinated=10 prevalence=10.0000%" in text
    assert "not a trained or unsupervised detector" in text
    assert "DETECTION NOT EVALUATED" in text
    assert "S=12 cells=30" in text
    assert "DiD@0=+3.0000" in text
    assert "evidence causal effect = observed-token logp(full)" in text
    assert "source dispersion [primary]" in text
    assert "ALL SAVED METRICS" not in text


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

    assert "ALL SAVED METRICS" in text
    assert "diagnostic_only" in text
    assert "raw(C/H)=n/a/n/a" in text
