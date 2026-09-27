"""Likelihood-ratio correctness and training-only transformation integrity."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from experiments.probabilistic_detection.model import (
    ConditionalGaussian, gaussian_log_density, shared_correlation_covariances,
)


COVARIANCES = ["full", "diagonal", "shared", "shared_correlation"]


def sample():
    random = np.random.default_rng(94)
    context = random.normal(size=(160, 2))
    labels = np.tile([0, 1], 80)
    observations = random.normal(size=(160, 3)) + labels[:, None] * .4 + context[:, :1]
    weights = random.uniform(.1, 2, len(labels))
    return context, observations, labels, weights


def test_gaussian_density_matches_independent_analytic_reference():
    covariance = np.array([[2., .6], [.6, 1.]])
    points = np.array([[1., 2.], [-3., 0.], [.1, -.2]])
    mean = np.array([.4, -.7])
    expected = multivariate_normal.logpdf(points, mean=mean, cov=covariance)
    actual = gaussian_log_density(points, mean, np.linalg.cholesky(covariance))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("covariance", COVARIANCES)
def test_weight_scale_cannot_change_regularization_or_scores(covariance):
    context, observations, labels, weights = sample()
    first = ConditionalGaussian(covariance=covariance).fit(context, observations, labels, weights)
    second = ConditionalGaussian(covariance=covariance).fit(context, observations, labels, weights * 731.)
    for name, values in first.predict_components(context, observations).items():
        np.testing.assert_allclose(values, second.predict_components(context, observations)[name], atol=1e-8)


@pytest.mark.parametrize("covariance", COVARIANCES)
def test_singular_channels_remain_finite(covariance):
    context, observations, labels, weights = sample()
    context[:, 1] = 1
    observations[:, 1] = observations[:, 0]
    observations[:, 2] = 0
    model = ConditionalGaussian(covariance=covariance, shrinkage=0).fit(context, observations, labels, weights)
    assert all(np.isfinite(value).all() for value in model.predict_components(context, observations).values())
    assert np.linalg.eigvalsh(model.covariances_).min() > 0


@pytest.mark.parametrize("covariance", COVARIANCES)
def test_swapping_labels_reverses_every_log_odds_component(covariance):
    context, observations, labels, weights = sample()
    original = ConditionalGaussian(covariance=covariance).fit(context, observations, labels, weights)
    swapped = ConditionalGaussian(covariance=covariance).fit(context, observations, 1 - labels, weights)
    first = original.predict_components(context, observations)
    second = swapped.predict_components(context, observations)
    for name in first:
        np.testing.assert_allclose(first[name], -second[name], atol=1e-8)


def test_prediction_does_not_fit_or_change_training_transforms(monkeypatch):
    context, observations, labels, weights = sample()
    model = ConditionalGaussian().fit(context, observations, labels, weights)
    saved_quantiles = model.observation_transform_.quantiles_.copy()
    saved_mean = model.context_scaler_.mean_.copy()
    expected = model.predict_components(context[:5], observations[:5])

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Prediction attempted to fit a transform")

    for transform in [model.observation_transform_, model.context_scaler_, model.context_spline_]:
        monkeypatch.setattr(transform, "fit", forbidden_fit)
    model.predict_components(context * 1000, observations * 1000)
    actual = model.predict_components(context[:5], observations[:5])
    np.testing.assert_array_equal(saved_quantiles, model.observation_transform_.quantiles_)
    np.testing.assert_array_equal(saved_mean, model.context_scaler_.mean_)
    for name in expected:
        np.testing.assert_array_equal(expected[name], actual[name])


def test_shared_and_diagonal_are_actual_covariance_controls():
    context, observations, labels, weights = sample()
    shared = ConditionalGaussian(covariance="shared").fit(context, observations, labels, weights)
    np.testing.assert_array_equal(shared.covariances_[0], shared.covariances_[1])
    diagonal = ConditionalGaussian(covariance="diagonal").fit(context, observations, labels, weights)
    for matrix in diagonal.covariances_:
        np.testing.assert_array_equal(matrix, np.diag(np.diag(matrix)))


def test_shared_correlation_pools_by_class_weight_and_preserves_variances():
    covariance = np.array([[[1., .8], [.8, 4.]], [[9., -1.2], [-1.2, 16.]]])
    pooled = shared_correlation_covariances(covariance, [1., 3.], shrinkage=.5)
    deviations = np.sqrt(np.diagonal(pooled, axis1=1, axis2=2))
    correlations = pooled / (deviations[:, :, None] * deviations[:, None, :])
    np.testing.assert_allclose(correlations[0], correlations[1], atol=1e-14)
    expected_off_diagonal = .5 * (.4 + 3 * -.1) / 4 / (1 + 1e-6)
    np.testing.assert_allclose(correlations[:, 0, 1], expected_off_diagonal, atol=1e-14)
    expected_variance = np.array([[1., 4.], [9., 16.]]) + np.array([[2.5e-6], [12.5e-6]])
    np.testing.assert_allclose(np.diagonal(pooled, axis1=1, axis2=2), expected_variance, atol=1e-14)
    assert np.linalg.eigvalsh(pooled).min() > 0


def test_shared_correlation_fitted_classes_have_common_correlation():
    context, observations, labels, weights = sample()
    observations[labels == 1] *= np.array([.2, 2., 1.])
    model = ConditionalGaussian(covariance="shared_correlation").fit(context, observations, labels, weights)
    deviations = np.sqrt(np.diagonal(model.covariances_, axis1=1, axis2=2))
    correlations = model.covariances_ / (deviations[:, :, None] * deviations[:, None, :])
    np.testing.assert_allclose(correlations[0], correlations[1], atol=1e-14)
    assert not np.allclose(deviations[0], deviations[1])
    assert all(np.isfinite(value).all() for value in model.predict_components(context, observations).values())


def test_one_class_is_rejected_instead_of_fabricated_density():
    context, observations, labels, weights = sample()
    with pytest.raises(ValueError, match="both binary classes"):
        ConditionalGaussian().fit(context, observations, labels * 0, weights)
