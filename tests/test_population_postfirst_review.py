"""CPU contracts for frozen post-first descriptive analysis."""

import numpy as np

from next_iteration.population_postfirst import fixed_scores, masks, weighted_metrics
from route_graph.metrics import binary_detection_metrics


def test_first_inclusive_and_post_first_masks_are_disjoint_complete_partition():
    labels = np.array([False, False, True, True, False, True])
    split = masks(labels)

    assert np.array_equal(split["through_first_error"], np.array([True, True, True, False, False, False]))
    assert np.array_equal(split["post_first_error"], np.array([False, False, False, True, True, True]))
    assert not np.any(split["through_first_error"] & split["post_first_error"])
    assert np.array_equal(split["all_tokens"], split["through_first_error"] | split["post_first_error"])
    no_error = masks(np.zeros(3, dtype=bool))
    assert no_error["through_first_error"].all()
    assert not no_error["post_first_error"].any()


def test_nine_fixed_scores_keep_declared_signs_and_source_balanced_metrics_match_legacy_helper():
    measured = {
        "base__entropy": np.array([.1, .2, .3, .4]),
        "base__margin": np.array([1., -2., 3., -4.]),
        "source_01__js": np.array([.01, .02, .03, .04]),
        "history_01__js": np.array([.11, .12, .13, .14]),
        "source_01__saved_logp_change": np.array([1., -2., 3., -4.]),
        "history_01__saved_logp_change": np.array([-1., 2., -3., 4.]),
        "source_permute__js": np.array([.21, .22, .23, .24]),
        "mlp_01__js": np.array([.31, .32, .33, .34]),
    }
    scores = fixed_scores(measured)
    assert list(scores) == [
        "entropy", "negative_margin", "source_small_js", "history_small_js",
        "source_saved_support", "history_saved_support", "history_minus_source_support",
        "source_permute_js", "mlp_small_js",
    ]
    assert np.array_equal(scores["negative_margin"], -measured["base__margin"])
    assert np.array_equal(scores["source_saved_support"], -measured["source_01__saved_logp_change"])
    assert np.array_equal(scores["history_saved_support"], -measured["history_01__saved_logp_change"])
    assert np.array_equal(scores["history_minus_source_support"], measured["source_01__saved_logp_change"] - measured["history_01__saved_logp_change"])

    y = np.array([False, True, False, True])
    source = np.array(["a", "a", "b", "b"])
    _, inverse = np.unique(source, return_inverse=True)
    expected = binary_detection_metrics(y, scores["entropy"], source, bootstrap=0, seed=20260912, source_balanced=True)
    actual = weighted_metrics(y, scores["entropy"], inverse)
    assert actual == {"auroc": expected["auroc"], "auprc": expected["auprc"], "bootstrap_replicates": 0}
