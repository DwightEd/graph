import numpy as np

from experiments.grounded_anchor_flow.evaluate import (
    bootstrap_metrics,
    paired_control,
    position_adjust,
)


def test_position_adjustment_removes_decile_center_without_labels():
    score = np.asarray([1.0, 3.0, 10.0, 14.0])
    relative = np.asarray([0.05, 0.05, 0.15, 0.15])
    adjusted = position_adjust(score, relative, np.ones(4, dtype=bool))

    np.testing.assert_allclose(np.median(adjusted[:2]), 0.0)
    np.testing.assert_allclose(np.median(adjusted[2:]), 0.0)


def test_paired_control_uses_the_same_tokens_and_source_draws():
    label = np.asarray([False, True, False, True, False, True])
    arrays = {
        "source_id": np.asarray(["a", "a", "b", "b", "c", "c"]),
        "primary": np.asarray([0.0, 1.0, 0.1, 0.9, 0.2, 0.8]),
        "primary__valid": np.asarray([True, True, True, True, True, False]),
        "control": np.asarray([1.0, 0.0, 0.2, 0.8, 0.1, 0.9]),
        "control__valid": np.asarray([True, True, True, True, False, True]),
    }

    result = paired_control("primary", "control", arrays, label, 50, 7)

    assert result["tokens"] == 4
    assert result["auroc_difference"] > 0
    assert result["bootstrap_successful"] == 50


def test_empty_bootstrap_returns_undefined_intervals():
    label = np.asarray([False, False])
    score = np.asarray([0.0, 1.0])
    source = np.asarray(["a", "b"])

    auroc, ap, successful = bootstrap_metrics(label, score, source, 10, 3)

    assert auroc == [None, None]
    assert ap == [None, None]
    assert successful == 0
