"""Scientifically consequential boundaries of the short-span audit."""

import numpy as np
import pytest

from experiments.short_span_audit.metrics import evaluate, normal_fpr_threshold


def block(scores, spans, *, source="s", answer="a", task="QA", pairs=()):
    arrays = {name: np.asarray(values, float) for name, values in scores.items()}
    length = len(next(iter(arrays.values())))
    labels = np.zeros(length, bool)
    for start, end in spans:
        labels[start:end] = True
    return {"record": dict(id=answer, source_id=source, dataset="RAGTruth", task=task, generator="m"),
            "tokens": length, "gold": np.asarray(spans, int).reshape(-1, 2), "scores": arrays,
            "alarms": {name: values >= .5 for name, values in arrays.items()},
            "pairs": list(pairs), "views": {"all_error": (labels, np.ones(length, bool))},
            "common_finite": np.ones(length, bool)}


def frozen(report, method="raw", task="QA"):
    return report["groups"][f"RAGTruth|{task}|m"]["methods"][method]["frozen"]


@pytest.mark.parametrize("budget, expected", [(0, 0), (.24, 0), (.25, 0), (.5, 2), (1, 4)])
def test_ties_never_exceed_normal_budget(budget, expected):
    scores = np.array([0., 0., 1., 1.])
    threshold = normal_fpr_threshold(scores, budget)
    alarms = scores >= threshold
    assert alarms.sum() == expected
    assert alarms.mean() <= budget
    if budget == 0:
        assert threshold == np.inf


def test_positive_budget_retains_errors_above_tied_normal_maximum():
    threshold = normal_fpr_threshold(np.ones(10), .05)
    assert threshold > 1
    assert 2 >= threshold


def test_float32_ties_preserve_float64_threshold_boundary():
    data = block({"raw": [2.] + [1.] * 100}, [(0, 1)])
    data["scores"]["raw"] = data["scores"]["raw"].astype(np.float32)
    report = evaluate([data], ["raw"], fpr_budgets=(.03,), bootstrap=0)
    point = report["groups"]["RAGTruth|QA|m"]["methods"]["raw"]
    curve = point["descriptive_test_normal_fpr_curve"][0]
    assert curve["achieved_normal_fpr"] == 0
    assert curve["bins"]["1-2"]["error"]["before_end_detected"] == 1


def test_missing_onset_is_unresolved_and_never_replaced_by_next_token():
    data = block({"raw": [np.nan, .1, .9, .1]}, [(0, 2)])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["bins"]["1-2"]["error"]
    assert result["missing_onsets"] == 1
    assert result["onset_recall"] is None
    assert result["before_end_detected"] == 0
    assert result["complete_miss"] == 0
    assert result["unresolved"] == 1
    assert result["before_end_recall_lower_bound"] == 0
    assert result["before_end_recall_upper_bound"] == 1


def test_alarm_after_end_does_not_count_and_detected_delay_is_explicit():
    data = block({"raw": [.1, .1, .9, .1, .1, .9, .1]}, [(0, 2), (4, 6)])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["bins"]["1-2"]["error"]
    assert result["before_end_detected"] == 1
    assert result["complete_miss"] == 1
    assert result["before_end_recall_lower_bound"] == .5
    assert result["detected_only_delay_mean"] == 1
    assert result["detected_only_delay_count"] == 1
    assert result["delay_population"] == "detected_only"


def test_missing_before_first_alarm_does_not_claim_exact_detection_delay():
    data = block({"raw": [np.nan, .9, .1]}, [(0, 2)])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["bins"]["1-2"]["error"]
    assert result["before_end_detected"] == 1
    assert result["detected_only_delay_mean"] == 1
    assert result["unambiguous_delay_count"] == 0
    assert result["unambiguous_detected_delay_mean"] is None


def test_no_normal_tokens_cannot_define_normal_fpr_operating_point():
    data = block({"raw": [.9]}, [(0, 1)])
    report = evaluate([data], ["raw"], fpr_budgets=(.05,), bootstrap=0)
    result = report["groups"]["RAGTruth|QA|m"]["methods"]["raw"]
    point = result["descriptive_test_normal_fpr_curve"][0]
    assert point["threshold_available"] is False
    assert point["achieved_normal_fpr"] is None


def test_each_span_gets_equal_weight_and_overlaps_merge_not_adjacent():
    data = block({"raw": [.9] + [.1] * 10}, [(0, 1), (1, 5), (3, 9)])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["bins"]["1-8"]["error"]
    assert result["spans"] == 2
    assert result["before_end_recall_lower_bound"] == .5
    assert len(report["span_rows"]) == 2


def test_matched_maximum_uses_whole_normal_span_and_keeps_pairs():
    pair = dict(pair_id="p", error_start=0, error_end=2, normal_start=3, normal_end=5)
    data = block({"raw": [.8, .2, .1, .1, .9]}, [(0, 2)], pairs=[pair])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    matched = frozen(report)["bins"]["1-2"]["matched"]["all"]
    assert matched["ranking"]["matched_span_max_auroc"] == 0
    assert matched["ranking"]["within_pair_win_rate"] == 0
    assert matched["normal"]["before_end_detected"] == 1
    assert matched["normal"]["onset_alarms"] == 0


def test_incomplete_pair_maxima_are_not_ranked():
    pair = dict(pair_id="p", error_start=0, error_end=2, normal_start=3, normal_end=5)
    data = block({"raw": [.8, .2, .1, np.nan, .9]}, [(0, 2)], pairs=[pair])
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    result = frozen(report)["bins"]["1-2"]["matched"]["all"]["ranking"]
    assert result["pairs"] == 1
    assert result["complete_pairs"] == 0
    assert result["matched_span_max_auroc"] is None


def test_recent_error_history_is_exactly_preceding_fifteen_tokens():
    pairs = [dict(pair_id="p1", error_start=2, error_end=3, normal_start=5, normal_end=6),
             dict(pair_id="p2", error_start=20, error_end=21, normal_start=40, normal_end=41)]
    scores = np.zeros(45)
    scores[5] = .9
    data = block({"raw": scores}, [(2, 3), (20, 21)], pairs=pairs)
    report = evaluate([data], ["raw"], fpr_budgets=(), bootstrap=0)
    matched = frozen(report)["bins"]["1-2"]["matched"]
    assert matched["recovery"]["normal"]["spans"] == 1
    assert matched["recovery"]["normal"]["before_end_detected"] == 1
    assert matched["clean_history"]["normal"]["spans"] == 1
    assert matched["clean_history"]["normal"]["before_end_detected"] == 0


def test_common_coverage_without_mutating_input_scores():
    data = block({"left": [np.nan, .9, .1], "right": [.9, .9, .1]}, [(0, 2)])
    report = evaluate([data], ["left", "right"], fpr_budgets=(), bootstrap=0)
    left = frozen(report, "left")["bins"]["1-2"]["error"]
    right = frozen(report, "right")["bins"]["1-2"]["error"]
    assert left["missing_onsets"] == right["missing_onsets"] == 1
    assert left["scored_tokens"] == right["scored_tokens"] == 1
    assert data["scores"]["right"][0] == .9


def test_descriptive_threshold_uses_whole_group_normals_not_matched_only():
    pair = dict(pair_id="p", error_start=0, error_end=1, normal_start=1, normal_end=2)
    data = block({"raw": [.7, .1, .2, .3, .9]}, [(0, 1)], pairs=[pair])
    report = evaluate([data], ["raw"], fpr_budgets=(.25,), bootstrap=0)
    method = report["groups"]["RAGTruth|QA|m"]["methods"]["raw"]
    point = method["descriptive_test_normal_fpr_curve"][0]
    assert point["normal_tokens"] == 4
    assert point["normal_alarms"] == 1
    assert point["achieved_normal_fpr"] == .25
    assert .3 < point["threshold"] < .7
    assert point["threshold_source"] == "test_normal_labels_descriptive_only"
    assert method["frozen"]["threshold_source"] == "saved_frozen_alarms"


def test_tasks_have_independent_thresholds():
    qa = block({"raw": [.7, .1, .2]}, [(0, 1)], task="QA")
    summary = block({"raw": [.7, .8, .9]}, [(0, 1)], task="Summary")
    report = evaluate([qa, summary], ["raw"], fpr_budgets=(.5,), bootstrap=0)
    points = [report["groups"][f"RAGTruth|{task}|m"]["methods"]["raw"]
              ["descriptive_test_normal_fpr_curve"][0] for task in ("QA", "Summary")]
    assert points[0]["bins"]["1-2"]["error"]["before_end_detected"] == 1
    assert points[1]["bins"]["1-2"]["error"]["before_end_detected"] == 0


def test_source_bootstrap_keeps_equal_span_estimand_and_clusters_answers():
    first = block({"left": [.9, .9, .9, .1], "right": [.1, .1, .1, .1]},
                  [(0, 1), (1, 2), (2, 3)], source="s1", answer="a1")
    second = block({"left": [.1, .1], "right": [.9, .1]},
                   [(0, 1)], source="s2", answer="a2")
    report = evaluate([first, second], ["left", "right"], fpr_budgets=(), bootstrap=200)
    difference = report["groups"]["RAGTruth|QA|m"]["comparisons"][0]["differences"][0]
    assert difference["delta"] == .5
    assert difference["spans"] == 4
    assert difference["sources"] == 2
    assert difference["ci95"] == [-1, 1]
    second["record"]["source_id"] = "s1"
    one_source = evaluate([first, second], ["left", "right"], fpr_budgets=(), bootstrap=200)
    difference = one_source["groups"]["RAGTruth|QA|m"]["comparisons"][0]["differences"][0]
    assert difference["sources"] == 1
    assert difference["bootstrap_valid"] == 0
    assert difference["ci95"] is None
