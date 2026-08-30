from experiments.attention_mechanism_audit.reporting import (
    KEY_METRICS,
    ONSET_METRICS,
    render_report,
    render_sample,
)


def _report():
    keys = {metric.key for metric in KEY_METRICS}
    keys.update(metric.key for metric in ONSET_METRICS)
    summary = {
        "correct_mean": 0.2,
        "hallucinated_mean": 0.3,
        "position_matched_source_equal_difference": 0.04,
        "ci95": [0.01, 0.07],
        "p_value": 0.02,
        "sources": 12,
        "matched_samples": 14,
        "matched_cells": 30,
        "covered_hallucinated_tokens": 8,
        "hallucinated_token_coverage": 0.8,
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
        "coverage": {"evaluated": 20, "eligible_qa": 20},
        "summaries": {key: dict(summary) for key in keys},
        "matched_onset": {key: dict(onset) for key in keys},
    }


def test_key_report_separates_audit_from_detection_and_prints_support_counts():
    text = render_report(_report())

    assert "20 responses | 100 tokens | 10 hallucinated (10.00%)" in text
    assert "evaluated 20/20 eligible QA" in text
    assert "matched 14 mixed responses, 12 sources, 30 position cells" in text
    assert "80.0% of hallucinated tokens" in text
    assert "Routing imbalance (%)" in text
    assert "Hallucination-span onset" in text
    assert "+3.0000 pp" in text
    assert "no detector score or threshold is evaluated" in text
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
        "p_value": None,
        "sources": 0,
        "matched_samples": 0,
        "matched_cells": 0,
        "covered_hallucinated_tokens": 0,
        "hallucinated_token_coverage": 0.0,
    }

    text = render_report(report, all_metrics=True)

    assert "All saved metrics" in text
    assert "diagnostic_only" in text
    assert "C=n/a H=n/a" in text


def test_sample_report_prints_onset_sources_and_heads():
    metric = {"all": 0.2, "correct": 0.1, "hallucinated": 0.4}
    record = {
        "sample_id": "42",
        "split": "train",
        "label": [0, 1],
        "hallucinated_tokens": 1,
        "hallucinated_fraction": 0.5,
        "summary": {
            name: dict(metric)
            for name in (
                "message_routing_drift_mean",
                "message_source_dispersion_mean",
                "head_role_disagreement_mean",
                "evidence_message_effect",
                "response_message_effect",
            )
        },
        "onsets": [
            {
                "start": 1,
                "token": "Paris",
                "span_text": "Paris",
                "changes_from_previous_token": {
                    "message_routing_drift_mean": 0.2,
                    "message_source_dispersion_mean": -0.1,
                    "evidence_message_effect": -0.5,
                },
                "evidence_effect": -0.3,
                "response_effect": 0.6,
                "full_margin": 0.4,
                "top_late_sources": [
                    {
                        "source_index": 3,
                        "role": "history",
                        "token": "is",
                        "late_retained_mass_over_total": 0.3,
                    }
                ],
                "top_late_head_routes": [
                    {"layer": 25, "head": 4, "role": "history", "edge_magnitude": 1.2}
                ],
            }
        ],
    }

    text = render_sample(record)

    assert "Sample 42 (train)" in text
    assert "Onset token 1" in text
    assert "top late-layer source routes" in text
    assert "L25 H04" in text
