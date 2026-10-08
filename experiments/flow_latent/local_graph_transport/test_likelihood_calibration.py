"""Reference weighting, ties and conditioning controls, independent of GT."""
import unittest

import numpy as np

from .likelihood_calibration import conditional_cdf, fit_conditional_cdf, shuffle_condition
from .unlabeled import equal_source_weights, fit_weighted_cdf, reference_rank


class LikelihoodCalibrationTests(unittest.TestCase):
    def test_constant_condition_is_global_weighted_midcdf(self):
        effect = np.array([-3., -3., 2., 7., 7.])
        sources = np.array([0, 0, 1, 1, 1])
        reference = fit_conditional_cdf(effect, np.full(5, -2.), sources)
        values, mass = fit_weighted_cdf(effect, equal_source_weights(sources))
        query = np.array([-9., -3., 0., 2., 7., 10.])
        expected = reference_rank(dict(x_values=values, x_cumulative=mass), 'x', query)
        actual = conditional_cdf(reference, query, np.zeros(len(query)))
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        np.testing.assert_allclose(actual, [0., .25, .5, 7/12, 5/6, 1.], atol=1e-15)

    def test_tied_condition_bins_and_within_bin_direction(self):
        likelihood = np.repeat([-10., -1.], 4)
        effect = np.tile([-2., 0., 0., 3.], 2)
        reference = fit_conditional_cdf(effect, likelihood, np.zeros(8))
        np.testing.assert_array_equal(reference['likelihood_edges'], [-1.])
        actual = conditional_cdf(reference, np.array([-2., 0., 3., -2., 0., 3.]),
                                 np.array([-100., -10., -10., -1., -1., 100.]))
        np.testing.assert_allclose(actual, [.125, .5, .875, .125, .5, .875])

    def test_replicating_one_sources_observations_preserves_reference(self):
        effect = np.array([-4., 2., -1., 3., 8.])
        likelihood = np.array([-9., -9., -9., -1., -1.])
        source = np.array([0, 0, 1, 1, 1])
        reference = fit_conditional_cdf(effect, likelihood, source)
        # Global source masses are .5 each: token weights .25,.25,1/6,1/6,1/6.
        # The -9 bin has total mass 2/3, hence sorted effect masses 3/8,1/4,3/8.
        # Recomputing equal-source weights inside that bin would be wrong.
        np.testing.assert_array_equal(reference['likelihood_edges'], [-1.])
        np.testing.assert_allclose(reference['bin_0_cumulative'], [0., 3/8, 5/8, 1.])
        np.testing.assert_allclose(conditional_cdf(reference, np.array([-4., -1., 2.]),
            np.full(3, -9.)), [3/16, .5, 13/16], atol=1e-15, rtol=0)
        indices = np.r_[np.tile([0, 1], 3), [2, 3, 4]]
        replicated = fit_conditional_cdf(effect[indices], likelihood[indices], source[indices])
        query = np.linspace(-10, 10, 21)
        for condition in (-9., -1.):
            np.testing.assert_allclose(conditional_cdf(reference, query, np.full(21, condition)),
                conditional_cdf(replicated, query, np.full(21, condition)), atol=1e-15, rtol=0)

    def test_shuffle_preserves_each_source_multiset_deterministically(self):
        source = np.repeat([0, 1], 20)
        values = np.arange(40, dtype=float)
        shuffled = shuffle_condition(values, source)
        np.testing.assert_array_equal(shuffled, shuffle_condition(values, source))
        self.assertFalse(np.array_equal(values, shuffled))
        for identity in (0, 1):
            np.testing.assert_array_equal(np.sort(shuffled[source == identity]), values[source == identity])


if __name__ == '__main__':
    unittest.main()
