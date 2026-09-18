import numpy as np

from experiments.ragtruth_flow.grounding_dynamics import (
    LinearAccumulator,
    CovarianceAccumulator,
    transition_xy,
)


def test_grounding_transition_predictors_keep_head_identity():
    routes = np.zeros((3, 2, 4, 4), dtype=float)
    routes[0, :, :, 3] = 1
    routes[1, :, :, 0] = 2
    routes[1, :, :, 3] = 3
    routes[2, :, :, 2] = 4
    x, y = transition_xy(routes)
    assert x.shape == (2, 2, 16)
    assert y.shape == (2, 2, 4)
    np.testing.assert_array_equal(x[0, 0, :4], np.ones(4))
    np.testing.assert_array_equal(x[0, 0, 4:8], np.full(4, 2.0))
    np.testing.assert_array_equal(y[0, 0], np.full(4, 2.0))


def test_linear_accumulator_recovers_simple_unlabelled_dynamics():
    x = np.arange(40, dtype=float).reshape(10, 4)
    y = x[:, :2] * 2 + 3
    accumulator = LinearAccumulator(4, 2)
    accumulator.add(x, y, 1.0)
    weight, intercept = accumulator.fit(1e-8)
    prediction = x @ weight + intercept
    np.testing.assert_allclose(prediction, y, atol=1e-5)


def test_covariance_reference_is_finite_with_ridge():
    values = np.array([[1., 2.], [2., 3.], [3., 4.]])
    accumulator = CovarianceAccumulator(2)
    accumulator.add(values, 1.0)
    mean, precision = accumulator.fit(1e-3)
    assert np.isfinite(mean).all()
    assert np.isfinite(precision).all()
