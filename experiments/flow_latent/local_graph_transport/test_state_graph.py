"""Exhaustive state enumeration and full physical-head gate checks on CPU."""
import itertools
import unittest

import numpy as np
from scipy.special import logsumexp

from .state_graph import (EPSILON, gated_weights, infer_states,
                           normalized_lag_weights)


def exhaustive(unary, weights, coupling, normalize=True):
    configurations = np.array(list(itertools.product((0, 1), repeat=len(unary))))
    probability = np.clip(unary, EPSILON, 1 - EPSILON)
    log_odds = np.log(probability) - np.log1p(-probability)
    normalized = normalized_lag_weights(weights, normalize=normalize)
    energy = configurations @ log_odds
    for token in range(len(unary)):
        for slot in range(min(token, weights.shape[1])):
            different = configurations[:, token] != configurations[:, token - slot - 1]
            energy -= coupling * normalized[token, slot] * different
    log_partition = logsumexp(energy)
    mass = np.exp(energy - log_partition)
    adjacent = np.empty((max(0, len(unary) - 1), 2, 2))
    for token in range(1, len(unary)):
        for previous, current in itertools.product((0, 1), repeat=2):
            selected = (configurations[:, token - 1] == previous) & (configurations[:, token] == current)
            adjacent[token - 1, previous, current] = mass[selected].sum()
    return configurations, mass, mass @ configurations, adjacent, log_partition


class StateGraphTests(unittest.TestCase):
    def test_zero_edges_and_zero_coupling_are_independent_unaries(self):
        unary = np.array([.03, .6, .9, .42])
        edges = np.zeros((4, 3))
        edges[1, 0], edges[3, 2] = .8, .5
        for weights, coupling in ((edges, 0.), (np.zeros_like(edges), 5.)):
            result = infer_states(unary, weights, coupling)
            np.testing.assert_array_equal(result['state1'], unary)
            expected = np.column_stack((1 - unary, unary))
            np.testing.assert_allclose(result['adjacent_joint'], expected[:-1, :, None] * expected[1:, None, :])

    def test_three_four_node_graphs_match_all_configuration_statistics(self):
        random = np.random.default_rng(42)
        for count, width in ((3, 2), (4, 3), (4, 8)):
            unary = random.uniform(.01, .99, count)
            edges = np.zeros((count, width))
            for token in range(count):
                edges[token, :min(token, width)] = random.uniform(0, 2, min(token, width))
            coupling = np.log(.993 / .007)
            configurations, mass, node, joint, log_z = exhaustive(unary, edges, coupling)
            result = infer_states(unary, edges, coupling)
            np.testing.assert_allclose(result['state1'], node, atol=2e-13, rtol=0)
            np.testing.assert_allclose(result['adjacent_joint'], joint, atol=2e-13, rtol=0)
            self.assertAlmostEqual(result['log_partition'], log_z, places=12)
            self.assertAlmostEqual(float(mass.sum()), 1., places=13)
            # Full-z statistics must agree with their returned node/joint marginals.
            self.assertAlmostEqual(float(result['state1'].sum()), float(mass @ configurations.sum(axis=1)), places=12)
            transitions = (configurations[:, 1:] != configurations[:, :-1]).sum(axis=1)
            self.assertAlmostEqual(float((result['enter'] + result['exit']).sum()), float(mass @ transitions), places=12)

    def test_chain_and_frontier_eviction_match_exhaustive(self):
        unary = np.array([.08, .8, .66, .04, .93, .7, .2, .45, .7, .15])
        for width in (1, 8):
            edges = np.zeros((len(unary), width))
            for token in range(len(unary)):
                edges[token, :min(token, width)] = np.linspace(.1, .5, min(token, width))
            _, _, node, joint, log_z = exhaustive(unary, edges, 2.7)
            result = infer_states(unary, edges, 2.7)
            np.testing.assert_allclose(result['state1'], node, atol=3e-13, rtol=0)
            np.testing.assert_allclose(result['adjacent_joint'], joint, atol=3e-13, rtol=0)
            self.assertAlmostEqual(result['log_partition'], log_z, places=12)
            np.testing.assert_allclose(joint.sum(axis=2)[:, 1], node[:-1], atol=3e-13)
            np.testing.assert_allclose(joint.sum(axis=1)[:, 1], node[1:], atol=3e-13)

    def test_unbudgeted_paper_chain_matches_original_transition_model(self):
        unary = np.array([.08, .82, .91, .25])
        edges = np.ones((4, 1))
        edges[0] = 0
        coupling = np.log(.993 / .007)
        configurations, mass, node, joint, log_z = exhaustive(unary, edges, coupling, normalize=False)
        result = infer_states(unary, edges, coupling, normalize=False)
        np.testing.assert_allclose(result['state1'], node, atol=2e-13, rtol=0)
        np.testing.assert_allclose(result['adjacent_joint'], joint, atol=2e-13, rtol=0)
        self.assertAlmostEqual(result['log_partition'], log_z, places=12)
        # Independent calculation uses a uniform initial state and p_stay=.993.
        probability = np.prod(np.where(configurations, unary, 1 - unary), axis=1) / 2
        for index in range(1, len(unary)):
            probability *= np.where(configurations[:, index] == configurations[:, index - 1], .993, .007)
        np.testing.assert_allclose(mass, probability / probability.sum(), atol=2e-13, rtol=0)
        np.testing.assert_array_equal(normalized_lag_weights(edges, normalize=False), edges)
        budgeted = infer_states(unary, edges, coupling)
        self.assertGreater(float(np.max(np.abs(budgeted['state1'] - node))), .01)

    def test_gates_preserve_head_identity_full_channels_and_signed_change(self):
        attention = np.zeros((4, 32, 8))
        attention[1:, :, 0] = .8
        heads = np.zeros((2, 5, 5, 32, 128))
        # Distinct physical heads: same last-channel coordinate, opposed and zero.
        heads[0, 1:, 4, 0, 127] = [1., 2., -2., -3.]
        heads[0, 1:, 1, 1, 63] = [1., 1., 1., 1.]
        weights, gates = gated_weights(attention, heads)
        np.testing.assert_array_equal(gates[1:, 0, 0], [1., 0., 1.])
        np.testing.assert_array_equal(gates[1:, 0, 1], [1., 1., 1.])
        self.assertTrue((gates[:, :, 2:] == 0).all())
        np.testing.assert_allclose(weights[1:, 0], [.05, .025, .05])
        self.assertTrue((weights[0] == 0).all())
        # Equal native/blocked additions cancel; no mean head vector is used.
        heads[:, :, 0, :, 3] += 11.
        unchanged, unchanged_gates = gated_weights(attention, heads)
        np.testing.assert_array_equal(unchanged, weights)
        np.testing.assert_array_equal(unchanged_gates, gates)


if __name__ == '__main__':
    unittest.main()
