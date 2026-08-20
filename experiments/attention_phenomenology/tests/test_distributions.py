import numpy as np

from experiments.attention_phenomenology.distributions import (
    close_compositions,
    dirichlet_logpdf,
    distribution_diagnostics,
    fit_dirichlet,
    fit_logistic_normal,
    logistic_normal_logpdf,
)


def test_dirichlet_fit_recovers_synthetic_mean():
    rng = np.random.default_rng(4)
    alpha = np.asarray([2.0, 5.0, 3.0, 1.5])
    values = rng.dirichlet(alpha, size=8000)

    model = fit_dirichlet(values)

    np.testing.assert_allclose(
        model.alpha / model.alpha.sum(),
        alpha / alpha.sum(),
        atol=0.015,
    )
    assert model.converged
    assert np.isfinite(dirichlet_logpdf(values[:10], model.alpha)).all()


def test_logistic_normal_density_is_finite():
    rng = np.random.default_rng(8)
    latent = rng.normal(size=(1000, 3))
    exp = np.exp(np.column_stack((latent, np.zeros(len(latent)))))
    values = exp / exp.sum(axis=1, keepdims=True)

    model = fit_logistic_normal(values)
    logpdf = logistic_normal_logpdf(values[:20], model)

    assert logpdf.shape == (20,)
    assert np.isfinite(logpdf).all()


def test_diagnostics_identify_well_specified_dirichlet():
    rng = np.random.default_rng(10)
    alpha = np.asarray([4.0, 2.0, 3.0, 1.0])
    fit = rng.dirichlet(alpha, size=5000)
    validation = rng.dirichlet(alpha, size=2000)

    _, _, metrics = distribution_diagnostics(
        fit,
        validation,
        simulation_rows=3000,
        seed=12,
    )

    assert metrics["dirichlet_converged"]
    assert abs(metrics["mean_l1_error"]) < 0.05
    assert metrics["positive_offdiagonal_covariance_fraction"] < 0.1
    assert metrics["nll_tail_probability_below_005_fraction"] < 0.09


def test_additive_smoothing_keeps_rows_in_open_simplex():
    values = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.3, 0.7]])

    closed = close_compositions(values, pseudocount=1e-4)

    np.testing.assert_allclose(closed.sum(axis=1), 1.0)
    assert np.all(closed > 0)


def test_diagnostics_reject_single_dirichlet_for_positive_dependence():
    rng = np.random.default_rng(2)
    covariance = np.asarray(
        [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 0.2]]
    )
    latent = rng.multivariate_normal(np.zeros(3), covariance, size=7000)
    exp = np.exp(np.column_stack((latent, np.zeros(len(latent)))))
    values = exp / exp.sum(axis=1, keepdims=True)

    _, _, metrics = distribution_diagnostics(
        values[:5000],
        values[5000:],
        simulation_rows=2000,
        seed=3,
    )

    assert metrics["positive_offdiagonal_covariance_fraction"] > 0.1
    assert metrics["dirichlet_minus_logistic_normal_nats"] < -0.2
