import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.unsupervised_token_graph.head_geometry.continuity_decomposition import (
    PHASES,
    decompose_difference,
    token_phases,
)


def block(source, gold, left, right):
    labels = np.zeros(len(left), dtype=bool)
    for start, end in gold:
        labels[start:end] = True
    return {
        "record": {"source_id": source},
        "tokens": len(left),
        "gold": gold,
        "views": {"all_error": (labels, np.ones(len(left), dtype=bool))},
        "scores": {"left": np.asarray(left, float), "right": np.asarray(right, float)},
    }


def test_exact_decomposition_and_continuation_with_ties_and_missing_positions():
    first = block(
        "a",
        [(1, 11)],
        [0, 0, 1, 2, 2, 1, 3, 4, 1, 0, 2, 3],
        [0, 1, 1, 0, 2, 3, 2, 2, 1, 0, 1, 2],
    )
    second = block("b", [(2, 4)], [0, 2, 1, 3, 2], [1, 2, 3, 2, 0])
    first["scores"]["left"][4] = np.nan
    result = decompose_difference([first, second], "left", "right", draws=30)
    labels = np.r_[first["views"]["all_error"][0], second["views"]["all_error"][0]]
    left = np.r_[first["scores"]["left"], second["scores"]["left"]]
    right = np.r_[first["scores"]["right"], second["scores"]["right"]]
    valid = np.isfinite(left) & np.isfinite(right)
    expected = roc_auc_score(labels[valid], left[valid]) - roc_auc_score(
        labels[valid], right[valid]
    )
    rows = result["partitions"]
    assert rows["all"]["delta_auroc"] == pytest.approx(expected)
    assert result["weighted_delta_sum"] == pytest.approx(expected)
    assert result["residual"] == pytest.approx(0, abs=1e-15)
    assert sum(rows[name]["positives"] for name in PHASES) == rows["all"]["positives"]
    assert rows["continuation"]["weighted_delta_auroc"] == pytest.approx(
        sum(rows[name]["weighted_delta_auroc"] for name in PHASES[1:])
    )
    assert rows["all"]["weighted_delta_source_bootstrap"]["valid"] == 30


def test_adjacent_annotations_have_separate_onsets_and_missing_tokens_keep_age():
    sample = block("a", [(1, 3), (3, 6)], np.arange(7), np.arange(7))
    sample["scores"]["left"][4] = np.nan
    np.testing.assert_array_equal(token_phases([sample]), [-1, 0, 1, 0, 1, 1, -1])
    result = decompose_difference([sample], "left", "right", draws=20)
    assert result["partitions"]["onset"]["positives"] == 2
    assert result["partitions"]["offset1_3"]["positives"] == 2
    assert result["partitions"]["offset4_7"]["left_auroc"] is None
    assert result["partitions"]["offset4_7"]["weighted_delta_auroc"] is None
    assert (
        result["partitions"]["all"]["weighted_delta_source_bootstrap"]["ci95"] is None
    )


def test_missing_class_never_becomes_zero_and_bootstrap_drops_invalid_draws():
    positive = block("positive", [(0, 3)], [1, 1, 2], [0, 1, 1])
    negative = block("negative", [], [0, 1, 1], [1, 1, 2])
    missing = decompose_difference([positive], "left", "right", draws=10)
    assert missing["partitions"]["all"]["delta_auroc"] is None
    assert missing["weighted_delta_sum"] is None
    assert missing["residual"] is None
    combined = decompose_difference([positive, negative], "left", "right", draws=100)
    count = combined["partitions"]["all"]["weighted_delta_source_bootstrap"]["valid"]
    assert 0 < count < 100
    repeat = decompose_difference([positive, negative], "left", "right", draws=100)
    assert combined == repeat


def test_bootstrap_absent_phase_has_zero_contribution_without_fabricating_auc():
    long = block("long", [(1, 11)], [1, *([2] * 10)], [1, *([0] * 10)])
    short = block("short", [(1, 2)], [1, 2], [1, 2])
    result = decompose_difference([long, short], "left", "right", draws=100)
    tail = result["partitions"]["offset8_plus"]
    interval = tail["weighted_delta_source_bootstrap"]
    assert interval["valid"] == 100
    assert interval["ci95"] == pytest.approx([0.0, 0.2])
    assert result["residual"] == pytest.approx(0, abs=1e-15)

    absent = decompose_difference([short], "left", "right", draws=0)
    assert absent["partitions"]["offset8_plus"]["left_auroc"] is None
    assert absent["partitions"]["offset8_plus"]["delta_auroc"] is None
