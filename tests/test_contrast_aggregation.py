"""CPU-only scientific checks for aggregation, ties, availability, and label isolation."""

import unittest

import numpy as np

from experiments.native_support.evidence_contrast.aggregation import (
    TOKEN_METHODS, WINDOW_CONTROLS, aggregate_scores, select_whole_units,
)


def observations():
    saved = {name: np.array([0., 2., 9., 3.]) for name in (*TOKEN_METHODS, *WINDOW_CONTROLS)}
    saved.update(target=np.arange(4), token_id=np.arange(4), risk=np.arange(4.))
    views = dict(answer_ids=list(range(4)), units=[dict(start=0, stop=2), dict(start=2, stop=4)])
    return saved, views


class AggregationTests(unittest.TestCase):
    def test_means_preserve_original_scores_and_reject_misalignment(self):
        saved, views = observations()
        scored = aggregate_scores(saved, views)
        for name in TOKEN_METHODS:
            np.testing.assert_array_equal(scored[name + "_unit_mean"], [1, 1, 6, 6])
            np.testing.assert_array_equal(scored[name], saved[name])
        np.testing.assert_array_equal(scored["risk"], saved["risk"])
        np.testing.assert_array_equal(scored["unit_end_target"], [1, 1, 3, 3])
        with self.assertRaisesRegex(ValueError, "partition"):
            aggregate_scores(saved, {**views, "units": [dict(start=0, stop=3), dict(start=2, stop=4)]})
        with self.assertRaisesRegex(ValueError, "target"):
            aggregate_scores({**saved, "target": np.arange(1, 5)}, views)


    def test_boundary_ties_never_split_units(self):
        np.testing.assert_array_equal(select_whole_units([3, 3, 1, 0]), [True, True, False, False])
        self.assertTrue(select_whole_units([0, 0, 0]).all())


if __name__ == "__main__":
    unittest.main()
