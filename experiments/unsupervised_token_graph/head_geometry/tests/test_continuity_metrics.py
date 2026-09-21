"""Count the actual annotation units and separate missingness from missed alarms."""

import numpy as np
import pytest

from experiments.unsupervised_token_graph.evaluate import Ranking
from experiments.unsupervised_token_graph.head_geometry.continuity_metrics import (
    annotation_summary,
    answer_rankings,
    merge_token_spans,
    offset_profile,
    paired_span_rows,
    span_rows,
    summarize_spans,
    within_answer_difference,
    within_answer_summary,
)


def block(scores, gold=(), response_id="a", source="s", labels=None):
    scores = np.asarray(scores, float)
    if labels is None:
        labels = np.zeros(len(scores), bool)
        for start, end in gold:
            labels[start:end] = True
    return {"record": {"id": response_id, "source_id": source}, "tokens": len(scores),
                "gold": np.asarray(gold, int).reshape(-1, 2),
                "views": {"all_error": (np.asarray(labels, bool), np.ones(len(scores), bool))},
                "scores": {"a": scores}, "alarms": {"a": scores > .5}}


def test_overlap_merges_but_adjacent_annotations_keep_their_onsets():
    sample = block([0, np.nan, .6, .7, .8, 0], [(1, 3), (2, 4), (4, 5)])
    np.testing.assert_array_equal(merge_token_spans(sample["gold"]), [[1, 4], [4, 5]])
    result = annotation_summary([sample], "a")
    assert result["input_token_spans"] == 3
    assert result["token_spans"] == 2
    assert result["positive_runs"] == 1
    assert result["error_tokens"] == 4
    assert result["onsets"] == 2
    assert result["continuation_fraction"] == .5
    assert result["scored_continuation_fraction"] == pytest.approx(2 / 3)
    assert result["scored_onsets"] == 1
    assert result["fully_scored_spans"] == 1
    assert result["length_histogram"] == {"1": 1, "3": 1}


def test_singletons_are_onsets_not_continuation_and_have_no_back_half():
    sample = block([.2, .8, .1], [(1, 2)])
    result = annotation_summary([sample], "a")
    assert result["continuation_tokens"] == 0
    assert result["continuation_fraction"] == 0
    profile = {row["group"]: row for row in offset_profile([sample], "a")["phases"]}
    assert profile["front"]["scored_tokens"] == 1
    assert profile["back"]["eligible_tokens"] == 0
    assert profile["back"]["score_mean"] is None
    assert profile["continuation"]["alarm_rate"] is None
    summary = summarize_spans(span_rows([sample], "a"))
    assert summary["onset_recall"] == 1
    assert summary["detected_only_delay_mean"] == 0


def test_within_answer_ranking_removes_answer_level_score_confounding():
    first = block([11, 10, 10, 10, 10], response_id="a", labels=[0, 1, 1, 1, 1])
    second = block([1, 1, 1, 1, 0], response_id="b", labels=[0, 0, 0, 0, 1])
    samples = [first, second]
    pooled = Ranking(np.concatenate([x["views"]["all_error"][0] for x in samples]),
                     np.concatenate([x["scores"]["a"] for x in samples])).measure()
    within = within_answer_summary(answer_rankings(samples, "a"))
    assert pooled["auroc"] == pytest.approx(.64)
    assert within["pair_weighted_auroc"] == 0
    assert within["macro_auroc"] == 0
    assert within["within_answer_pairs"] == 8


def test_pair_weighting_differs_from_macro_and_keeps_exclusion_denominators():
    large = block([.1, .2, .8, .9], response_id="a", labels=[0, 0, 1, 1])
    small = block([.9, .1], response_id="b", labels=[0, 1])
    normal = block([.1, .2], response_id="c", labels=[0, 0])
    missing = block([.1, np.nan], response_id="d", labels=[0, 1])
    result = within_answer_summary(answer_rankings([large, small, normal, missing], "a"))
    assert result["macro_auroc"] == .5
    assert result["pair_weighted_auroc"] == .8
    assert result["included_answers"] == 2
    assert result["excluded_answers"] == 2
    assert result["excluded_scored_tokens"] == 3
    assert result["scored_all_normal_answers"] == 2
    assert result["mixed_answers_lost_to_missing"] == 1


def test_missing_onset_does_not_become_a_false_onset_or_known_miss():
    sample = block([np.nan, .2, .9, .1, .1, np.nan], [(0, 3), (3, 5), (5, 6)])
    rows = span_rows([sample], "a")
    assert rows[0]["onset_alarm"] is None
    assert rows[0]["first_observed_alarm_delay"] == 2
    assert rows[0]["missing_before_alarm"] == 1
    summary = summarize_spans(rows)
    assert summary["observed_onsets"] == 1
    assert summary["missing_onsets"] == 2
    assert summary["completely_observed_missed"] == 1
    assert summary["incomplete_without_alarm"] == 1
    assert summary["no_score_spans"] == 1
    assert summary["detected_only_delay_count"] == 1
    assert summary["unambiguous_delay_count"] == 0
    assert summary["unambiguous_detected_delay_mean"] is None


def test_paired_spans_preserve_same_length_and_normal_false_alarm_rate():
    sample = block([.2, .7, .9, .1, .2, .3], [(0, 3)])
    pair = {"pair_id": "a:0", "span_index": 0, "error_start": 0, "error_end": 3,
                "normal_start": 3, "normal_end": 6, "length": 3, "previous_error": False}
    rows = paired_span_rows(sample, "a", [pair])
    assert [row["side"] for row in rows] == ["error", "normal"]
    assert [row["length"] for row in rows] == [3, 3]
    assert rows[0]["token_alarm_rate"] == pytest.approx(2 / 3)
    assert rows[1]["token_alarm_rate"] == 0
    assert rows[0]["first_observed_alarm_delay"] == 1


def test_offset_measurement_cannot_see_later_score_and_reports_attrition():
    sample = block([.2, .7, .9, .3], [(0, 3), (3, 4)])
    original = offset_profile([sample], "a")["offsets"]
    sample["scores"]["a"][2] = 100
    changed = offset_profile([sample], "a")["offsets"]
    assert original[0] == changed[0]
    assert original[1] == changed[1]
    assert original[0]["eligible_tokens"] == 2
    assert original[1]["eligible_tokens"] == 1
    assert changed[2]["score_mean"] == 100


def test_bootstrap_uses_sources_and_common_finite_tokens():
    first = block([.1, .9, np.nan], response_id="a", source="one", labels=[0, 1, 1])
    second = block([.1, .9], response_id="b", source="one", labels=[0, 1])
    first["scores"]["b"] = np.array([.9, .1, .5])
    second["scores"]["b"] = np.array([.9, .1])
    same_source = within_answer_difference([first, second], "all_error", "a", "b", 100)
    assert same_source["sources"] == 1
    assert same_source["bootstrap_valid"] == 0
    assert same_source["coverage"]["scored_tokens"] == 4
    assert same_source["delta"]["pair_weighted_auroc"] == 1
    second["record"]["source_id"] = "two"
    independent = within_answer_difference([first, second], "all_error", "a", "b", 100)
    assert independent["bootstrap_valid"] == 100
    assert independent["ci95"][0][0] == 1
    assert independent["ci95"][1][0] == 1


def test_no_positive_annotations_does_not_create_empty_span_statistics():
    sample = block([.1, .2])
    result = annotation_summary([sample], "a")
    assert result["token_spans"] == 0
    assert result["fully_scored_spans"] == 0
    assert result["continuation_fraction"] is None
    assert result["length_quantiles"] is None
    assert summarize_spans(span_rows([sample], "a"))["onset_recall"] is None
