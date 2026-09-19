import numpy as np

from experiments.unsupervised_token_graph.structured_compatibility.evaluation import (
    threshold_metrics,
    transition_views,
)


def test_transition_views_fix_previous_label_without_future_information():
    error = np.array([False, True, True, False, False])
    sentence = np.array([True, False, False, False, True])
    jump = np.array([0., 2., 1., 3., 0.])

    views = transition_views(
        error,
        sentence,
        jump,
        high_transition=1.5,
    )
    _, previous_zero = views["previous_gold_0"]
    _, previous_one = views["previous_gold_1"]

    assert previous_zero.tolist() == [True, True, False, False, True]
    assert previous_one.tolist() == [False, False, True, True, False]


def test_threshold_metrics_uses_only_scoped_finite_scores():
    blocks = [{
        "transition": {
            "previous_gold_0": (
                np.array([False, True, False]),
                np.array([True, True, True]),
            )
        },
        "scores": {
            "compatibility": np.array([0., 2., np.nan]),
        },
        "alarms": {
            "compatibility": np.array([False, True, False]),
        },
    }]
    result = threshold_metrics(
        blocks,
        "transition",
        "previous_gold_0",
        "compatibility",
    )
    assert result["recall"] == 1.
    assert result["fpr"] == 0.
