"""Scientific invariants of annotation-free address and decision readouts."""

import unittest

import numpy as np
import torch

from experiments.role_free_flow.features import (candidate_ids, lens_readouts,
    routing_layer, source_balanced_percentile)
from experiments.role_free_flow.run import candidate_lens


class RoleFreeFlowTests(unittest.TestCase):
    def test_same_mass_different_addresses_are_detected(self):
        attention = np.zeros((1, 3, 6), dtype=np.float32)
        attention[0, 0, :2] = [.5, 0]
        attention[0, 1, :2] = [0, .5]
        attention[0, 2, :2] = [.25, .25]
        measured = routing_layer(attention, 3, np.zeros(6, bool))
        np.testing.assert_allclose(measured[0, :2, 0], [.5, .5])
        self.assertAlmostEqual(float(measured[0, 1, 1]), np.log(2), places=6)
        self.assertAlmostEqual(float(measured[0, 2, 1]), 0, places=6)

    def test_routing_does_not_read_future_or_special_keys(self):
        rng = np.random.default_rng(4)
        weights = rng.random((2, 5, 8)).astype(np.float32)
        special = np.array([True, False, False, False, False, False, False, False])
        expected = routing_layer(weights, 3, special)
        changed = weights.copy()
        changed[:, :, 0] = 1e4
        for step in range(5):
            changed[:, step, 3 + step:] = 1e4
        np.testing.assert_allclose(routing_layer(changed, 3, special), expected)

    def test_no_duplicate_candidates_or_manual_truth_direction(self):
        ids, valid = candidate_ids(np.array([[1, 2], [1, 2]]), np.array([1, 3]))
        np.testing.assert_array_equal(valid, [[True, True, False], [True, True, True]])
        logits = np.tile(np.array([[2., 0., 2.], [2., 0., -1.]]), (32, 1, 1))
        out = lens_readouts(logits, valid, ids, np.array([1, 3]))
        np.testing.assert_allclose(out['chosen_margin_final'], [2, -3])
        np.testing.assert_allclose(out['final_top1_settle_layer'], 1)
        np.testing.assert_allclose(out['late_candidate_js'], 0, atol=1e-8)

    def test_final_hidden_is_not_normalized_twice_and_query_precedes_target(self):
        hidden = np.zeros((33, 4, 2), dtype=np.float32)
        hidden[:, 1] = [3., 4.]
        hidden[:, 2] = [99., 99.]
        ids = np.array([[0, 1]])
        logits = candidate_lens(hidden, 2, ids, torch.eye(2), np.ones(2), 0.)
        np.testing.assert_allclose(logits[-1, 0], [3, 4])
        np.testing.assert_allclose(logits[0, 0], np.array([3, 4]) / np.sqrt(12.5), rtol=1e-6)

    def test_reference_is_source_balanced_and_ties_are_conservative(self):
        values = source_balanced_percentile(np.array([0., 0., 0., 10.]),
                                           np.array(['a', 'a', 'a', 'b']), np.array([0., 1., 11.]))
        np.testing.assert_allclose(values, [0., .5, 1.])


if __name__ == '__main__':
    unittest.main()
