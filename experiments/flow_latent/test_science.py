"""Scientific contracts: graph direction, density and label-free fitting."""
import unittest

import numpy as np
from scipy.stats import multivariate_normal
from .flow import matched_mappings, rooted_paths, coordinate_projection
from .model import component_log_density, fit_mixture, conditional_log_density, factor_posterior


class ScientificContracts(unittest.TestCase):
    def test_paths_equal_finite_inverse(self):
        source = np.arange(12, dtype=float).reshape(3, 2, 2)
        weights = np.zeros((3, 2, 3))
        weights[1, :, 0] = .3
        weights[2, :, 0] = .2
        weights[2, :, 1] = .4
        result = rooted_paths(source, weights)
        for head in range(2):
            operator = weights[:, head]
            expected = np.linalg.solve(np.eye(3) - .5 * operator, source[:, head])
            np.testing.assert_allclose(result[:, head], expected)
            np.testing.assert_allclose(np.linalg.eigvals(operator), 0)

    def test_rewire_preserves_domains_tokens_lags_and_future(self):
        tokens = [4, 4, 4, 5, 4, 4, 4, 4, 4, 5]
        prompt = 5
        source = [True, True, True, False, False]
        mappings, movable = matched_mappings(tokens, prompt, source, heads=2)
        domain = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        for target, mapping in enumerate(mappings):
            query = prompt + target - 1
            for head in range(2):
                np.testing.assert_array_equal(np.sort(mapping[head]), np.arange(len(tokens)))
                np.testing.assert_array_equal(np.asarray(tokens)[mapping[head]], tokens)
                np.testing.assert_array_equal(domain[mapping[head]], domain)
                np.testing.assert_array_equal(mapping[head, query:], np.arange(query, len(tokens)))
                lag = np.floor(np.log2(query - np.arange(query + 1) + 1))
                np.testing.assert_array_equal(lag[mapping[head, :query + 1]], lag)
        self.assertTrue(any(mask.any() for mask in movable))

    def test_density_matches_independent_gaussian(self):
        nodes = np.array([[.2, -.8], [.1, .3]])
        context = np.ones((2, 1))
        coefficient = np.array([[.1, .2], [.3, .4]])
        covariance = np.array([[2., .3], [.3, 1.]])
        actual = component_log_density(nodes, context, [coefficient], [covariance])[:, 0]
        expected = multivariate_normal.logpdf(nodes, mean=coefficient.sum(0), cov=covariance)
        np.testing.assert_allclose(actual, expected)

    def test_em_state_posterior_finite_and_predictive(self):
        generator = np.random.default_rng(4)
        context = generator.normal(size=(160, 2))
        nodes = context @ generator.normal(size=(2, 8)) + .3 * generator.normal(size=(160, 8))
        fitted = fit_mixture(nodes, context, seed=42, iterations=8)
        density, posterior = conditional_log_density(nodes, context, fitted)
        shifted, _ = conditional_log_density(nodes + 8, context, fitted)
        self.assertTrue(np.isfinite(density).all())
        np.testing.assert_allclose(posterior.sum(1), 1)
        self.assertGreater(density.mean(), shifted.mean())
        self.assertGreater(fitted['weights'].min(), 0)
        objective = np.array([row['map_objective'] for row in fitted['trace']])
        self.assertGreaterEqual(np.diff(objective).min(), -1e-7)
        mean, covariance = factor_posterior(nodes, context, fitted)
        self.assertEqual(mean.shape, (len(nodes), 4, 4))
        for index, loading in enumerate(fitted['loadings']):
            expected = np.linalg.inv(np.eye(4) + loading.T @ loading / fitted['noise'][index])
            np.testing.assert_allclose(covariance[index], expected, atol=1e-10)

    def test_fixed_coordinate_projection(self):
        axes = coordinate_projection()
        np.testing.assert_allclose(axes.T @ axes, np.eye(4), atol=3e-7)


if __name__ == '__main__':
    unittest.main()
