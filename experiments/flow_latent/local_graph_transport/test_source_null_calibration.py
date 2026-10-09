"""Scientific contracts for a physical-effect null, independent of error labels."""
import unittest

import numpy as np

from .likelihood_calibration import conditional_cdf, fit_conditional_cdf
from .source_null_calibration import balanced_center, bounded_center
from .unlabeled import graph_fields


class SourceNullCalibrationTests(unittest.TestCase):
    def test_skewed_nulls_have_same_coordinate(self):
        ranks = np.array([.95, .3, 0., 1.])
        for center in (bounded_center, balanced_center):
            np.testing.assert_array_equal(center(ranks, ranks), np.full(4, .5))

    def test_bounded_center_matches_hand_calculated_skewed_references(self):
        actual = bounded_center(np.array([0., .05, .1, .95, 1.]), .95)
        np.testing.assert_allclose(actual, [0., 1/38, 1/19, .5, 10/19],
                                   atol=1e-15, rtol=0)
        actual = bounded_center(np.array([0., .3, .65, 1.]), .3)
        np.testing.assert_allclose(actual, [2/7, .5, .75, 1.], atol=1e-15, rtol=0)

    def test_bounded_center_has_no_tail_amplification(self):
        ranks = np.linspace(0., 1., 1001)[:, None]
        nulls = np.array([0., .000001, .3, .5, .95, .999999, 1.])
        values = bounded_center(ranks, nulls)
        changes = np.diff(values, axis=0)
        self.assertTrue(np.all(changes >= 0.))
        self.assertTrue(np.all(changes <= .001 + 5e-16))
        self.assertTrue(np.all(values >= 0.))
        self.assertTrue(np.all(values <= 1.))
        np.testing.assert_allclose(values[500, 3], .5, atol=1e-15, rtol=0)

    def test_balanced_center_expands_the_small_reference_tail(self):
        ranks = np.array([0., .475, .95, .975, 1.])
        np.testing.assert_allclose(balanced_center(ranks, .95),
                                   [0., .25, .5, .75, 1.], atol=1e-15, rtol=0)
        small_tail = balanced_center(np.array([.95, .951]), .95)
        np.testing.assert_allclose(small_tail, [.5, .51], atol=1e-15, rtol=0)
        bounded_tail = bounded_center(np.array([.95, .951]), .95)
        self.assertLess(float(bounded_tail[1] - bounded_tail[0]), .001)

    def test_endpoint_nulls_and_plateaus_are_neutral_without_division(self):
        ranks = np.array([0., 0., .2, 1., 1., .8])
        nulls = np.array([0., 0., 0., 1., 1., 1.])
        with np.errstate(divide='raise', invalid='raise'):
            for center in (bounded_center, balanced_center):
                np.testing.assert_allclose(center(ranks, nulls),
                                           [.5, .5, .6, .5, .5, .4],
                                           atol=1e-15, rtol=0)

    def test_symmetric_reference_recovers_original_rank(self):
        ranks = np.array([0., .125, .3, .5, .875, 1.])
        for center in (bounded_center, balanced_center):
            np.testing.assert_allclose(center(ranks, .5), ranks, atol=1e-15, rtol=0)

    def test_broadcasting_permutation_and_signed_sides(self):
        ranks = np.array([[0., .1, .4], [.5, .5, .5], [1., .9, .6]])
        nulls = np.array([.5, .3, .8])
        permutation = np.array([2, 0, 1])
        for center in (bounded_center, balanced_center):
            values = center(ranks, nulls)
            np.testing.assert_allclose(center(ranks[:, permutation], nulls[permutation]),
                                       values[:, permutation], atol=1e-15, rtol=0)
            self.assertTrue(np.all(values[ranks <= nulls] <= .5))
            self.assertTrue(np.all(values[ranks >= nulls] >= .5))

    def test_native_likelihood_bins_no_longer_move_the_measured_null(self):
        effect = np.r_[np.arange(-6., 0.), np.arange(1., 15.),
                       np.arange(-19., 0.), 1.]
        likelihood = np.repeat([-10., -.001], 20)
        reference = fit_conditional_cdf(effect, likelihood, np.zeros(40))
        conditions = np.array([-10., -.001])
        nulls = conditional_cdf(reference, np.zeros(2), conditions)
        np.testing.assert_allclose(nulls, [.3, .95], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(bounded_center(nulls, nulls), [.5, .5])
        for sign in (-1., 1.):
            ranks = conditional_cdf(reference, np.full(2, sign), conditions)
            adjusted = bounded_center(ranks, nulls)
            self.assertTrue(np.all(sign * (adjusted - .5) > 0.))

    def test_zero_atom_retains_real_empirical_jumps(self):
        effect = np.array([-2., -1., 0., 0., 2.])
        reference = fit_conditional_cdf(effect, np.full(5, -2.), np.zeros(5))
        query = np.array([-1e-12, 0., 1e-12])
        ranks = conditional_cdf(reference, query, np.full(3, -2.))
        np.testing.assert_allclose(ranks, [.4, .6, .8], atol=1e-15, rtol=0)
        nulls = conditional_cdf(reference, np.zeros(3), np.full(3, -2.))
        np.testing.assert_allclose(bounded_center(ranks, nulls),
                                   [1/3, .5, 2/3], atol=1e-15, rtol=0)
        np.testing.assert_allclose(balanced_center(ranks, nulls),
                                   [1/3, .5, .75], atol=1e-15, rtol=0)

    def test_all_zero_reference_is_resolved_only_at_its_atom(self):
        reference = fit_conditional_cdf(np.zeros(4), np.full(4, -1.), np.zeros(4))
        ranks = conditional_cdf(reference, np.array([-1., 0., 1.]), np.full(3, -1.))
        nulls = conditional_cdf(reference, np.zeros(3), np.full(3, -1.))
        for center in (bounded_center, balanced_center):
            np.testing.assert_array_equal(center(ranks, nulls), [0., .5, 1.])

    def test_equal_source_reference_survives_replication_with_hand_numbers(self):
        effect = np.array([-4., 2., -1., 3., 8.])
        sources = np.array([0, 0, 1, 1, 1])
        conditions = np.full(5, -2.)
        query = np.array([-5., -4., -1., 0., 2., 8., 10.])
        query_conditions = np.full(7, -2.)
        repeat = np.r_[np.tile([0, 1], 3), [2, 3, 4]]
        for indices in (np.arange(5), repeat):
            reference = fit_conditional_cdf(effect[indices], conditions[indices],
                                            sources[indices])
            ranks = conditional_cdf(reference, query, query_conditions)
            nulls = conditional_cdf(reference, np.zeros(7), query_conditions)
            np.testing.assert_allclose(nulls, np.full(7, 5/12), atol=1e-15, rtol=0)
            np.testing.assert_allclose(bounded_center(ranks, nulls),
                                       [1/7, .25, 3/7, .5, 17/28, 13/14, 1.],
                                       atol=1e-15, rtol=0)

    def test_neutral_source_preserves_individual_route_through_original_graph(self):
        centered = bounded_center(np.array([.3, .95]), np.array([.3, .95]))
        unary = .375 * (centered + centered) + .25 * np.array([0., 1.])
        np.testing.assert_array_equal(unary, [.375, .625])
        result, _ = graph_fields(unary, dict(native=np.array([[0.], [.4]])))
        # Two-node solve: the mean stays .5 and the difference contracts by 1.4.
        np.testing.assert_allclose(result['native_huber'], [23/56, 33/56],
                                   atol=1e-8, rtol=0)
        constant = np.full(2, .425)
        result, _ = graph_fields(constant, dict(native=np.array([[0.], [.4]])))
        np.testing.assert_array_equal(result['native_huber'], constant)


if __name__ == '__main__':
    unittest.main()
