"""Synthetic scientific checks for KL propagation; no natural labels are read."""
import unittest
from decimal import Decimal, localcontext

import numpy as np
from scipy.optimize import root
from scipy.special import expit, logit
import torch

from .kl_graph import (
    apply_laplacian, kl_gradient, kl_objective, kl_objective_change, kl_smooth_field,
    kl_smooth_normalized, prepare_graph, quadratic_smooth_normalized,
)
from .smooth import smooth_field


def random_graph(count, seed=7):
    """Sample all original strict-past lag<=8 edges with uneven raw weights."""
    random = np.random.default_rng(seed)
    sources = []
    targets = []
    for target in range(1, count):
        for source in range(max(0, target - 8), target):
            sources.append(source)
            targets.append(target)
    weights = random.lognormal(0, 1.5, len(sources))
    return np.array(sources, dtype=np.int64), np.array(targets, dtype=np.int64), weights


def dense_laplacian(graph):
    """Independent edge outer-product construction for dense reference checks."""
    count = len(graph['degree'])
    incidence = np.zeros((len(graph['sources']), count), dtype=np.float64)
    for row, (source, target) in enumerate(zip(graph['sources'], graph['targets'])):
        incidence[row, source] = -1
        incidence[row, target] = 1
    return incidence.T @ (graph['weights'][:, None] * incidence)


def decimal_objective(field, unary, graph, gamma=2.):
    """Independent high-precision objective using exact float input values."""
    field = [Decimal.from_float(float(value)) for value in field]
    unary = [Decimal.from_float(float(value)) for value in unary]
    one = Decimal(1)
    result = Decimal(0)
    for value, target in zip(field, unary):
        result += value * (value / target).ln()
        result += (one - value) * ((one - value) / (one - target)).ln()
    for source, target, weight in zip(graph['sources'], graph['targets'], graph['weights']):
        difference = field[target] - field[source]
        result += Decimal.from_float(gamma / 2 * float(weight)) * difference * difference
    return result


class KLGraphTest(unittest.TestCase):
    def test_normalization_and_edge_quadratic_identity(self):
        sources, targets, raw = random_graph(19)
        graph = prepare_graph(sources, targets, raw, 19)
        raw_degree = np.bincount(sources, raw, minlength=19)
        raw_degree += np.bincount(targets, raw, minlength=19)
        expected = raw / np.maximum(1, np.maximum(raw_degree[sources], raw_degree[targets]))
        np.testing.assert_allclose(graph['weights'], expected, rtol=0, atol=1e-16)
        self.assertTrue(np.all(graph['degree'] <= 1 + 1e-15))

        field = np.random.default_rng(9).normal(size=19)
        laplacian = dense_laplacian(graph)
        np.testing.assert_allclose(apply_laplacian(field, graph), laplacian @ field, atol=1e-15)
        edge_energy = np.dot(graph['weights'], np.square(field[targets] - field[sources]))
        self.assertAlmostEqual(field @ laplacian @ field, edge_energy, places=13)

    def test_kl_gradient_matches_independent_autograd_and_hessian(self):
        sources, targets, raw = random_graph(7)
        graph = prepare_graph(sources, targets, raw, 7)
        unary = np.array([.11, .87, .35, .69, .24, .48, .91])
        field = np.array([.31, .71, .39, .57, .29, .47, .79])
        tensor = torch.tensor(field, dtype=torch.float64, requires_grad=True)
        target_tensor = torch.tensor(unary, dtype=torch.float64)
        divergence = tensor * (tensor / target_tensor).log()
        divergence += (1 - tensor) * ((1 - tensor) / (1 - target_tensor)).log()
        difference = tensor[targets] - tensor[sources]
        objective = divergence.sum() + (torch.tensor(graph['weights']) * difference.square()).sum()
        gradient = torch.autograd.grad(objective, tensor, create_graph=True)[0]
        hessian = torch.stack([torch.autograd.grad(value, tensor, retain_graph=True)[0]
                               for value in gradient])
        expected_hessian = np.diag(1 / (field * (1 - field))) + 2 * dense_laplacian(graph)
        np.testing.assert_allclose(gradient.detach().numpy(),
                                   kl_gradient(field, logit(unary), graph), atol=2e-15)
        np.testing.assert_allclose(hessian.numpy(), expected_hessian, atol=2e-15)
        self.assertAlmostEqual(float(objective.detach()), kl_objective(field, unary, graph), places=14)

    def test_newton_matches_independent_dense_root(self):
        sources, targets, raw = random_graph(17, seed=12)
        graph = prepare_graph(sources, targets, raw, 17)
        unary = np.random.default_rng(4).uniform(.02, .98, 17)
        laplacian = dense_laplacian(graph)
        # Solve for log-odds instead of score coordinates to stay interior.
        def gradient(coordinates):
            return coordinates - logit(unary) + 2 * laplacian @ expit(coordinates)

        def jacobian(coordinates):
            field = expit(coordinates)
            return np.eye(len(field)) + 2 * laplacian * (field * (1 - field))[None, :]

        reference = root(gradient, logit(unary), jac=jacobian, method='hybr', options={'xtol': 1e-11})
        self.assertTrue(reference.success, reference.message)
        result = kl_smooth_normalized(unary, graph)
        np.testing.assert_allclose(result['solution'], expit(reference.x), rtol=0, atol=2e-10)
        self.assertLessEqual(result['max_gradient'], 1e-9)

    def test_objective_change_bregman_identity_and_tiny_step_precision(self):
        sources, targets, raw = random_graph(17)
        graph = prepare_graph(sources, targets, raw, 17)
        random = np.random.default_rng(5)
        unary = random.uniform(.02, .98, 17)
        field = random.uniform(.1, .9, 17)
        candidate = random.uniform(.1, .9, 17)
        gradient = kl_gradient(field, logit(unary), graph)
        expected = kl_objective(candidate, unary, graph) - kl_objective(field, unary, graph)
        self.assertAlmostEqual(kl_objective_change(field, candidate, gradient, graph), expected, places=13)

        # At exact unary, the leading change is its KL Hessian quadratic form.
        tiny_candidate = unary + 1e-10 * random.normal(size=17)
        difference = tiny_candidate - unary
        hessian = np.diag(1 / (unary * (1 - unary))) + 2 * dense_laplacian(graph)
        gradient = kl_gradient(unary, logit(unary), graph)
        quadratic = gradient @ difference + .5 * difference @ hessian @ difference
        stable_change = kl_objective_change(unary, tiny_candidate, gradient, graph)
        self.assertLess(abs(stable_change - quadratic), 1e-24)
        with localcontext() as context:
            context.prec = 60
            exact_change = decimal_objective(tiny_candidate, unary, graph)
            exact_change -= decimal_objective(unary, unary, graph)
        self.assertLess(abs(stable_change - float(exact_change)), 1e-23)

    def test_long_graph_and_tail_stationarity_without_score_clipping(self):
        for count in (32, 100, 700):
            for seed in range(8):
                sources, targets, raw = random_graph(count, seed)
                graph = prepare_graph(sources, targets, raw, count)
                unary = expit(np.random.default_rng(seed + 23).normal(0, 3, count))
                for gamma in (.5, 2):
                    result = kl_smooth_normalized(unary, graph, gamma)
                    self.assertLessEqual(result['max_gradient'], 1e-9)
                    self.assertTrue(np.all(np.abs(result['logit_shift']) <= gamma * graph['degree'] + 1e-9))

    def test_random_stationarity_maximum_principle_and_logit_bound(self):
        for count, seed in ((4, 19), (23, 24), (80, 28)):
            sources, targets, raw = random_graph(count, seed)
            graph = prepare_graph(sources, targets, raw, count)
            unary = np.random.default_rng(seed).uniform(.005, .995, count)
            result = kl_smooth_normalized(unary, graph)
            self.assertLessEqual(result['max_gradient'], 1e-9)
            self.assertGreaterEqual(result['solution'].min(), unary.min())
            self.assertLessEqual(result['solution'].max(), unary.max())
            self.assertTrue(np.all(np.abs(result['logit_shift']) <= 2 * graph['degree'] + 1e-9))
            self.assertGreater(result['min_endpoint_distance'], 0)
            self.assertLessEqual(result['objective'], kl_objective(unary, unary, graph))

    def test_zero_edges_zero_gamma_and_isolated_nodes(self):
        unary = np.array([.11, .93, .49])
        empty = np.empty(0, dtype=np.int64)
        zero_graph = prepare_graph(empty, empty, np.empty(0), len(unary))
        np.testing.assert_array_equal(kl_smooth_normalized(unary, zero_graph)['solution'], unary)
        graph = prepare_graph(np.array([0]), np.array([1]), np.array([3.]), len(unary))
        np.testing.assert_array_equal(kl_smooth_normalized(unary, graph, gamma=0)['solution'], unary)
        self.assertEqual(kl_smooth_normalized(unary, graph)['solution'][2], unary[2])

    def test_constant_field_is_preserved_exactly(self):
        sources, targets, raw = random_graph(13)
        graph = prepare_graph(sources, targets, raw, 13)
        for level in (.013, .5, .991):
            unary = np.full(13, level)
            result = kl_smooth_normalized(unary, graph)
            np.testing.assert_array_equal(result['solution'], unary)
            self.assertEqual(result['iterations'], 0)

    def test_complement_symmetry(self):
        sources, targets, raw = random_graph(21)
        graph = prepare_graph(sources, targets, raw, 21)
        unary = np.random.default_rng(2).uniform(.01, .99, 21)
        forward = kl_smooth_normalized(unary, graph)
        complement = kl_smooth_normalized(1 - unary, graph)
        np.testing.assert_allclose(complement['solution'], 1 - forward['solution'], atol=2e-10)
        self.assertAlmostEqual(forward['objective'], complement['objective'], places=12)

    def test_reversing_indices_preserves_symmetric_graph_solution(self):
        count = 16
        sources, targets, raw = random_graph(count)
        unary = np.random.default_rng(3).uniform(.02, .98, count)
        result = kl_smooth_field(unary, sources, targets, raw)
        reversed_sources = count - 1 - targets
        reversed_targets = count - 1 - sources
        reversed_result = kl_smooth_field(unary[::-1], reversed_sources, reversed_targets, raw)
        np.testing.assert_allclose(reversed_result['solution'][::-1], result['solution'], atol=2e-10)

    def test_exact_score_diffusion_reproduces_old_optimizer(self):
        sources, targets, raw = random_graph(48)
        graph = prepare_graph(sources, targets, raw, 48)
        unary = np.random.default_rng(11).uniform(.001, .999, 48)
        exact = quadratic_smooth_normalized(unary, graph)
        old = smooth_field(torch.tensor(unary), torch.tensor(sources), torch.tensor(targets),
                           torch.tensor(raw))
        np.testing.assert_allclose(exact['solution'], old['solution'].numpy(), rtol=0, atol=1e-8)
        dense = np.linalg.solve(np.eye(48) + .5 * dense_laplacian(graph), unary)
        np.testing.assert_allclose(exact['solution'], dense, rtol=0, atol=1e-15)
        self.assertLess(exact['max_gradient'], 1e-15)

    def test_gamma_two_has_old_midpoint_linear_response(self):
        sources, targets, raw = random_graph(15)
        graph = prepare_graph(sources, targets, raw, 15)
        direction = np.random.default_rng(5).normal(size=15)
        perturbation = 1e-5
        unary = .5 + perturbation * direction
        result = kl_smooth_normalized(unary, graph)
        response = (result['solution'] - .5) / perturbation
        reference = np.linalg.solve(np.eye(15) + .5 * dense_laplacian(graph), direction)
        np.testing.assert_allclose(response, reference, rtol=0, atol=1e-7)

    def test_logit_control_dense_identity_and_nonuniversal_peak_retention(self):
        graph = prepare_graph(np.array([0]), np.array([1]), np.array([1.]), 2)
        unary = np.array([.999, 1e-10])
        control = quadratic_smooth_normalized(unary, graph, coordinate='logit')
        expected = expit(np.linalg.solve(np.eye(2) + .5 * dense_laplacian(graph), logit(unary)))
        np.testing.assert_allclose(control['solution'], expected, atol=1e-15)
        # A tiny neighbor can reverse even an extreme positive logit peak.
        self.assertLess(control['solution'][0], .5)
        self.assertLess(control['solution'][0],
                        quadratic_smooth_normalized(unary, graph)['solution'][0])

    def test_unsupported_endpoint_and_failed_convergence_are_explicit(self):
        graph = prepare_graph(np.array([0]), np.array([1]), np.array([1.]), 2)
        for unary in (np.array([0., .5]), np.array([.5, 1.])):
            with self.assertRaisesRegex(ValueError, 'strict interior'):
                kl_smooth_normalized(unary, graph)
        with self.assertRaisesRegex(RuntimeError, 'did not converge'):
            kl_smooth_normalized(np.array([.1, .9]), graph, max_iterations=0)


if __name__ == '__main__':
    unittest.main()
