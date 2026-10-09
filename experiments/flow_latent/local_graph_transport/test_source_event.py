"""Synthetic probability-object checks; no natural references or labels."""
import unittest
from decimal import Decimal, localcontext

import numpy as np
from scipy.integrate import quad

from .source_event import (
    bernoulli_fisher_angle, probability_difference, source_event_effects,
    uniform_view_logmix,
)


class SourceEventTest(unittest.TestCase):
    def test_uniform_mixture_matches_probability_and_logaddexp(self):
        local = np.array([-500., -30., -4., -.2, 0.])
        full = np.array([-510., -29., -.5, -.9, 0.])
        actual = uniform_view_logmix(local, full)
        expected = np.logaddexp(local, full) - np.log(2)
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=5e-15)
        np.testing.assert_allclose(np.exp(actual), .5 * (np.exp(local) + np.exp(full)), rtol=1e-13)
        self.assertEqual(actual[-1], 0.)
        self.assertTrue(np.all(actual <= 0))

    def test_tiny_negative_mixture_has_high_precision_complement(self):
        tiny = -1e-30
        actual = float(uniform_view_logmix(np.array(0.), np.array(tiny)))
        with localcontext() as context:
            context.prec = 80
            exact = ((Decimal(1) + Decimal.from_float(tiny).exp()) / 2).ln()
        self.assertEqual(actual, float(exact))
        self.assertLess(actual, 0)
        self.assertGreater(float(np.sqrt(-np.expm1(actual))), 0)
        difference = float(probability_difference(np.array(tiny), np.array(0.)))
        self.assertEqual(difference, -np.expm1(tiny))
        self.assertGreater(difference, 0)

    def test_finite_endpoint_values_and_bounds(self):
        logp = np.array([0., -np.log(2), -1000., -2000.])
        angles = bernoulli_fisher_angle(logp)
        self.assertEqual(angles[0], np.pi)
        self.assertAlmostEqual(angles[1], np.pi / 2, places=15)
        self.assertGreater(angles[2], 0)
        self.assertEqual(angles[3], 0)
        result = source_event_effects(logp, logp, logp[::-1], logp[::-1])
        for value in result.values():
            self.assertTrue(np.isfinite(value).all())
        self.assertEqual(result['fisher_effect'][0], -np.pi)
        self.assertEqual(result['fisher_effect'][-1], np.pi)
        self.assertTrue(np.all(np.abs(result['fisher_effect']) <= np.pi))

    def test_equal_conditions_give_exact_zero(self):
        local = np.array([0., -1e-30, -1., -1000.])
        full = np.array([0., -.2, -50., -1100.])
        result = source_event_effects(local, full, local, full)
        for field in ('fisher_effect', 'logp_effect', 'probability_effect'):
            np.testing.assert_array_equal(result[field], np.zeros_like(local))

    def test_source_swap_negates_all_effects(self):
        values = np.random.default_rng(4).uniform(-30, 0, (4, 31))
        result = source_event_effects(*values)
        swapped = source_event_effects(values[2], values[3], values[0], values[1])
        for field in ('fisher_effect', 'logp_effect', 'probability_effect'):
            np.testing.assert_array_equal(swapped[field], -result[field])

    def test_uniform_mixing_is_invariant_to_each_view_swap(self):
        values = np.random.default_rng(6).uniform(-15, 0, (4, 21))
        result = source_event_effects(*values)
        present_swap = source_event_effects(values[1], values[0], values[2], values[3])
        absent_swap = source_event_effects(values[0], values[1], values[3], values[2])
        for field in result:
            np.testing.assert_array_equal(result[field], present_swap[field])
            np.testing.assert_array_equal(result[field], absent_swap[field])

    def test_observable_effect_signs_agree(self):
        values = np.random.default_rng(11).uniform(-50, -.001, (4, 200))
        result = source_event_effects(*values)
        expected_sign = np.sign(result['logp_effect'])
        np.testing.assert_array_equal(np.sign(result['fisher_effect']), expected_sign)
        np.testing.assert_array_equal(np.sign(result['probability_effect']), expected_sign)

    def test_extreme_logp_preserves_fisher_root_beyond_probability_underflow(self):
        result = source_event_effects(-1000., -1000., -999., -999.)
        self.assertGreater(float(result['fisher_effect']), 0)
        self.assertAlmostEqual(float(result['fisher_effect']) / np.exp(-500),
                               2 * np.expm1(.5), places=14)
        self.assertEqual(float(result['logp_effect']), 1.)
        self.assertEqual(float(result['present_probability']), 0.)
        self.assertEqual(float(result['absent_probability']), 0.)
        self.assertEqual(float(result['probability_effect']), 0.)

    def test_fisher_derivative_and_integrated_metric(self):
        probability = np.array([.1, .25, .5, .75, .9])
        step = 1e-6
        derivative = (bernoulli_fisher_angle(np.log(probability + step))
                      - bernoulli_fisher_angle(np.log(probability - step))) / (2 * step)
        expected = 1 / np.sqrt(probability * (1 - probability))
        np.testing.assert_allclose(derivative, expected, rtol=2e-10, atol=2e-10)
        integral = quad(lambda value: 1 / np.sqrt(value * (1 - value)), .1, .9)[0]
        observed = float(bernoulli_fisher_angle(np.log(.9)) - bernoulli_fisher_angle(np.log(.1)))
        self.assertAlmostEqual(observed, integral, places=12)

    def test_positive_or_nonfinite_native_logp_is_rejected(self):
        for invalid in (1e-300, .1, np.nan, np.inf, -np.inf):
            with self.assertRaisesRegex(ValueError, 'finite and nonpositive'):
                source_event_effects(invalid, -1., -1., -1.)
        with self.assertRaises(ValueError):
            source_event_effects(np.array([-1., -2.]), np.array([-1.]), -1., -1.)


if __name__ == '__main__':
    unittest.main()
