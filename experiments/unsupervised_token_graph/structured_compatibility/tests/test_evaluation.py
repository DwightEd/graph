import numpy as np

from experiments.unsupervised_token_graph.structured_compatibility.evaluation import (
    binding_arrays,
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


def test_binding_arrays_backfills_prompt_length_from_frozen_record():
    class Saved(dict):
        @property
        def files(self):
            return list(self)

    saved = Saved(
        token_ids=np.array([1, 2, 3, 4]),
        offsets=np.array([[0, 1], [1, 2]]),
    )
    arrays = binding_arrays(saved, {"prompt_length": 2})
    assert int(arrays["prompt_length"]) == 2
    np.testing.assert_array_equal(arrays["token_ids"], [1, 2, 3, 4])
    np.testing.assert_array_equal(arrays["offsets"], [[0, 1], [1, 2]])


def test_binding_arrays_omits_empty_offsets_for_exact_recovery():
    saved = {
        "token_ids": np.array([1, 2, 3]),
        "offsets": np.empty((0, 2), dtype=int),
    }
    arrays = binding_arrays(saved, {"prompt_length": 1})
    assert "offsets" not in arrays
