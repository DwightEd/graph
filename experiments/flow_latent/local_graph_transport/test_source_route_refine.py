"""Scientific checks for signal cancellation, scale controls and robust continuity."""
import unittest

import numpy as np
import torch

from .source_route_refine import (edge_reference, fit_fusion_reference, score_answer,
                                  unaries, weighted_median)
from .smooth import smooth_field
from .unlabeled import fit_reference, score_answer as old_score_answer


class SourceRouteRefineTests(unittest.TestCase):
    def setUp(self):
        self.values = dict(source_local=np.array([0., 1., 2., 3.]),
                           source_full=np.array([3., 2., 1., 0.]),
                           raw_route=np.array([0., 1., 2., 3.]))
        self.scalar = fit_reference(*self.values.values(), np.zeros(4))
        rank = {name: np.array([.125, .375, .625, .875]) for name in self.values}
        self.fusion = fit_fusion_reference(rank['source_local'], rank['source_full'],
                                           np.zeros(4))
        self.attention = np.zeros((4, 32, 8))
        self.attention[1:, :, 0] = .7

    def test_max_retains_disagreement_mean_cancels(self):
        fields, ranks = unaries(self.scalar, self.fusion, self.values)
        self.assertTrue(np.all(ranks['source_mean'] == .5))
        self.assertGreater(ranks['source_max'][0], ranks['source_mean'][0])
        self.assertGreater(fields['union'][0], fields['mean_cdf'][0])

    def test_exact_old_native_reproduction(self):
        new = score_answer(self.scalar, self.fusion, .1, self.values, self.attention)
        old = old_score_answer(self.scalar, *self.values.values(), self.attention)
        np.testing.assert_array_equal(new['scores']['old_unary'], old['scores']['source_route_unary'])
        np.testing.assert_array_equal(new['scores']['old_native'], old['scores']['source_route_native_huber'])

    def test_robust_preserves_any_spike_not_only_errors(self):
        unary = torch.tensor([0., 1., 0.], dtype=torch.float64)
        senders, targets = torch.tensor([0, 1]), torch.tensor([1, 2])
        weights = torch.ones(2, dtype=torch.float64)
        quadratic = smooth_field(unary, senders, targets, weights)['solution']
        robust = smooth_field(unary, senders, targets, weights, huber_delta=.1)
        self.assertGreater(float(robust['solution'][1]), float(quadratic[1]))
        self.assertLessEqual(robust['max_actual_change'], .05 + 2e-8)
        self.assertGreater(float((robust['solution'][1] - robust['solution'][0]).abs()), .1)

    def test_source_equal_edge_median_and_zero_failure(self):
        self.assertEqual(weighted_median([.1, .1, .8], [.25, .25, .5]), .1)
        with self.assertRaisesRegex(ValueError, 'zero'):
            weighted_median([0., .2], [.7, .3])

    def test_full_answer_graph_and_zero_attention(self):
        fields, _ = unaries(self.scalar, self.fusion, self.values)
        difference, weight = edge_reference(fields['old'], self.attention)
        self.assertEqual(len(difference), 3)
        self.assertTrue((weight > 0).all())
        result = score_answer(self.scalar, self.fusion, .1, self.values,
                              np.zeros_like(self.attention))
        for name in ('old', 'union', 'mean_cdf'):
            np.testing.assert_array_equal(result['scores'][f'{name}_unary'],
                                          result['scores'][f'{name}_robust'])
            self.assertEqual(result['diagnostics'][f'{name}_robust']['linear_mass_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
