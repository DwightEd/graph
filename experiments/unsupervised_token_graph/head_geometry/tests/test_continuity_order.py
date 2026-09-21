import numpy as np

from experiments.unsupervised_token_graph.head_geometry.continuity_order import (
    order_audit,
    ordering_arrays,
    smooth_run,
)


def test_joint_permutation_preserves_pointwise_ranking_and_missing_gaps():
    values = np.array([0., .4, 1., np.nan, .8, .3, .1])
    labels = np.array([False, False, True, True, True, False, False])
    blocks = [{"scores": {"raw": values}, "views": {"all_error": (labels, ~np.isnan(values))}}]
    report = order_audit(blocks, "raw", 3, permutations=5)
    for row in report["draws"]:
        assert row["raw"] == report["observed"]["raw"]
        assert row["tokens"] == 6
    _, _, smooth, _ = ordering_arrays(blocks, "raw", 3)
    assert smooth[3] == .8  # A missing observation resets the trailing window.
    np.testing.assert_array_equal(values, [0., .4, 1., np.nan, .8, .3, .1])


def test_smoothing_is_causal_and_zero_is_observed():
    values = np.array([1., 0., 2., 100.])
    result = smooth_run(values, 2)
    np.testing.assert_allclose(result[:3], smooth_run(values[:3], 2))
    np.testing.assert_allclose(result, [1., .5, 1., 51.])


def test_single_class_has_no_ranking_gain():
    block = {"scores": {"raw": np.zeros(4)}, "views": {"all_error": (np.ones(4, bool), np.ones(4, bool))}}
    report = order_audit([block], "raw", 2, permutations=2)
    assert report["observed"]["auroc_gain"] is None
    assert report["null"]["auroc_gain"]["mean"] is None
