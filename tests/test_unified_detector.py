"""Source-scale integrity, constrained matrix inference, and immutable CPU cache scoring."""


import numpy as np

from experiments.native_support.unified.calibration import fit_distribution, transform


def test_calibration_balances_sources_and_does_not_turn_ties_into_different_scores():
    fitted = fit_distribution([0., 0., 1.], ["long", "long", "short"])
    repeated = fit_distribution([0.] * 20 + [1.], ["long"] * 20 + ["short"])
    query = np.array([-.1, 0., .5, 1., 1.1])
    np.testing.assert_allclose(transform(query, fitted), [0., .25, .5, .75, 1.])
    np.testing.assert_allclose(transform(query, fitted), transform(query, repeated))
