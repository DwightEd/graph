"""Failures at onset, delayed alarms, censored exits and paired false positives."""

import numpy as np
from test_short_span_metrics import block, frozen

from experiments.short_span_audit.metrics import evaluate


def test_deadlines_count_missed_and_short_spans_and_bound_missing_data():
    data = block({"raw": [0.1, 0.9, 0.1, 0.1, np.nan, 0.1, 0.1]}, [(0, 2), (3, 4), (4, 6)])
    result = frozen(evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0))["bins"]["1-8"]["error"]
    assert result["spans"] == 3
    assert result["recall_by_0_lower"] == 0
    assert result["recall_by_0_upper"] == 1 / 3
    assert result["recall_by_1_lower"] == 1 / 3
    assert result["recall_by_1_upper"] == 2 / 3
    # A short miss never becomes a success through its normal recovery region.
    assert result["recall_by_7_lower"] == 1 / 3


def test_recovery_stops_at_clear_missing_next_error_or_answer_end():
    data = block(
        {"raw": [0.9, 0.9, 0.1, 0.9, 0.9, np.nan, 0.9, 0.9, 0.9]}, [(0, 1), (3, 4), (6, 7), (8, 9)]
    )
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    rows = report["span_rows"]
    assert [row["alarm_tail_stop"] for row in rows] == [
        "clear",
        "missing",
        "next_error",
        "answer_end",
    ]
    assert [row["alarm_tail_lower_bound"] for row in rows] == [1, 1, 1, 0]
    assert [row["alarm_tail_exact"] for row in rows] == [True, False, False, False]
    # Adjacent annotations have no recovery observations; do not invent a zero-length exit.
    assert rows[-1]["post_end_tokens"] == 0


def test_late_only_alarm_is_a_failure_and_horizon_tail_is_censored():
    data = block({"raw": [0.1] + [0.9] * 20 + [0.1]}, [(0, 1)])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    span = report["span_rows"][0]
    result = frozen(report)["bins"]["1-2"]["error"]
    assert result["before_end_detected"] == 0
    assert result["late_only_alarm_spans"] == 1
    assert span["alarm_tail_lower_bound"] == 15
    assert span["alarm_tail_stop"] == "horizon"
    assert result["spans_ending_in_alarm"] == 0


def test_all_normal_answer_false_alarms_use_answer_denominator_and_missing_bounds():
    data = [
        block({"raw": [0.9, 0.1]}, [], answer="a"),
        block({"raw": [0.1, np.nan]}, [], answer="b"),
        block({"raw": [0.9, 0.1]}, [(0, 1)], answer="c"),
    ]
    report = evaluate(data, ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["answer_alarms"]
    assert result["all_normal_answers"] == 2
    assert result["normal_answer_alarm_rate_lower"] == 0.5
    assert result["normal_answer_alarm_rate_upper"] == 1
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (1, 0, 1, 3)


def test_matched_gain_disappears_when_normal_false_alarms_increase_equally():
    pair = {"pair_id": "p", "error_start": 0, "error_end": 1, "normal_start": 2, "normal_end": 3}
    data = [
        block(
            {"left": [0.9, 0.1, 0.9], "right": [0.1, 0.1, 0.1]},
            [(0, 1)],
            pairs=[pair],
            source=f"s{i}",
            answer=f"a{i}",
        )
        for i in range(2)
    ]
    report = evaluate(data, ["left", "right"], fpr_budgets=(), bootstrap=20)
    group = report["groups"]["RAGTruth|QA|m"]
    differences = {d["metric"]: d for c in group["comparisons"] for d in c["differences"]}
    assert differences["matched_error_recall"]["delta"] == 1
    assert differences["matched_normal_fpr"]["delta"] == 1
    assert differences["matched_recall_minus_fpr"]["delta"] == 0
    assert differences["matched_recall_minus_fpr"]["ci95"] == [0, 0]
    confusion = frozen(report, "left")["bins"]["1-8"]["matched"]["all"]["confusion"]
    assert (confusion["tp"], confusion["fn"], confusion["fp"], confusion["tn"]) == (2, 0, 2, 0)


def test_long_span_bins_are_disjoint_and_retain_legacy_total():
    data = block({"raw": [0.1] * 30}, [(0, 10), (10, 30)])
    bins = frozen(evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0))["bins"]
    assert bins["9-16"]["error"]["spans"] == 1
    assert bins["17+"]["error"]["spans"] == 1
    assert bins["9+"]["error"]["spans"] == 2
