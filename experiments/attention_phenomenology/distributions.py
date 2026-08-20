"""Parametric models and diagnostics for attention compositions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import kstest


@dataclass(frozen=True)
class DirichletModel:
    alpha: np.ndarray
    converged: bool
    iterations: int


@dataclass(frozen=True)
class LogisticNormalModel:
    mean: np.ndarray
    covariance: np.ndarray


def close_compositions(values: np.ndarray, pseudocount: float) -> np.ndarray:
    """Map non-negative rows to the open simplex with additive smoothing."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("composition values must be a two-dimensional matrix")
    matrix = np.clip(matrix, 0.0, None)
    row_sum = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("every composition row must have positive total mass")
    matrix = matrix / row_sum
    if pseudocount > 0:
        matrix = (matrix + pseudocount) / (
            1.0 + pseudocount * matrix.shape[1]
        )
    return matrix


def _dirichlet_initial_alpha(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0)
    variance = values.var(axis=0, ddof=1)
    candidate = mean * (1.0 - mean) / np.maximum(variance, 1e-12) - 1.0
    candidate = candidate[np.isfinite(candidate) & (candidate > 0)]
    concentration = float(np.median(candidate)) if len(candidate) else float(len(mean))
    concentration = np.clip(concentration, 1e-2, 1e6)
    return np.clip(mean * concentration, 1e-3, 1e6)


def fit_dirichlet(
    values: np.ndarray,
    *,
    pseudocount: float = 1e-4,
    maximum_iterations: int = 500,
) -> DirichletModel:
    """Maximum-likelihood fit using a log-parameter L-BFGS optimization."""

    matrix = close_compositions(values, pseudocount)
    rows = len(matrix)
    sum_log = np.log(matrix).sum(axis=0)
    initial = _dirichlet_initial_alpha(matrix)

    def objective(log_alpha: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = np.exp(log_alpha)
        alpha_sum = alpha.sum()
        log_likelihood = (
            rows * (gammaln(alpha_sum) - gammaln(alpha).sum())
            + ((alpha - 1.0) * sum_log).sum()
        )
        gradient_alpha = (
            -rows * digamma(alpha_sum)
            + rows * digamma(alpha)
            - sum_log
        )
        return float(-log_likelihood), gradient_alpha * alpha

    result = minimize(
        objective,
        np.log(initial),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-12.0, 16.0)] * matrix.shape[1],
        options={"maxiter": maximum_iterations, "ftol": 1e-10},
    )
    return DirichletModel(
        alpha=np.exp(result.x),
        converged=bool(result.success),
        iterations=int(result.nit),
    )


def dirichlet_logpdf(
    values: np.ndarray,
    alpha: np.ndarray,
    *,
    pseudocount: float = 1e-4,
) -> np.ndarray:
    matrix = close_compositions(values, pseudocount)
    parameter = np.asarray(alpha, dtype=np.float64)
    normalizer = gammaln(parameter.sum()) - gammaln(parameter).sum()
    return normalizer + ((parameter - 1.0) * np.log(matrix)).sum(axis=1)


def fit_logistic_normal(
    values: np.ndarray,
    *,
    pseudocount: float = 1e-4,
    covariance_shrinkage: float = 0.05,
) -> LogisticNormalModel:
    """Fit a Gaussian in additive-log-ratio coordinates."""

    matrix = close_compositions(values, pseudocount)
    transformed = np.log(matrix[:, :-1] / matrix[:, -1, None])
    mean = transformed.mean(axis=0)
    covariance = np.atleast_2d(np.cov(transformed, rowvar=False, ddof=1))
    dimensions = covariance.shape[0]
    average_variance = float(np.trace(covariance) / max(dimensions, 1))
    covariance = (
        (1.0 - covariance_shrinkage) * covariance
        + covariance_shrinkage * average_variance * np.eye(dimensions)
    )
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.maximum(eigenvalue, 1e-6)
    covariance = (eigenvector * eigenvalue) @ eigenvector.T
    return LogisticNormalModel(mean=mean, covariance=covariance)


def logistic_normal_logpdf(
    values: np.ndarray,
    model: LogisticNormalModel,
    *,
    pseudocount: float = 1e-4,
) -> np.ndarray:
    matrix = close_compositions(values, pseudocount)
    transformed = np.log(matrix[:, :-1] / matrix[:, -1, None])
    centered = transformed - model.mean
    sign, log_determinant = np.linalg.slogdet(model.covariance)
    if sign <= 0:
        raise ValueError("logistic-normal covariance must be positive definite")
    solution = np.linalg.solve(model.covariance, centered.T).T
    dimensions = centered.shape[1]
    gaussian = -0.5 * (
        dimensions * np.log(2.0 * np.pi)
        + log_determinant
        + (centered * solution).sum(axis=1)
    )
    jacobian = -np.log(matrix).sum(axis=1)
    return gaussian + jacobian


def dirichlet_moments(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parameter = np.asarray(alpha, dtype=np.float64)
    concentration = parameter.sum()
    mean = parameter / concentration
    covariance = -np.outer(mean, mean) / (concentration + 1.0)
    diagonal = mean * (1.0 - mean) / (concentration + 1.0)
    np.fill_diagonal(covariance, diagonal)
    return mean, covariance


def _positive_covariance_statistics(covariance: np.ndarray) -> tuple[float, float]:
    mask = ~np.eye(covariance.shape[0], dtype=bool)
    off_diagonal = covariance[mask]
    positive = np.clip(off_diagonal, 0.0, None)
    absolute = np.abs(off_diagonal)
    return (
        float(np.mean(off_diagonal > 1e-8)),
        float(positive.sum() / max(absolute.sum(), 1e-12)),
    )


def _moment_concentration_dispersion(values: np.ndarray) -> float:
    mean = values.mean(axis=0)
    variance = values.var(axis=0, ddof=1)
    estimate = mean * (1.0 - mean) / np.maximum(variance, 1e-12) - 1.0
    estimate = estimate[np.isfinite(estimate) & (estimate > 0)]
    if len(estimate) < 2:
        return float("nan")
    return float(np.std(estimate) / max(np.mean(estimate), 1e-12))


def distribution_diagnostics(
    fit_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    pseudocount: float = 1e-4,
    simulation_rows: int = 4096,
    seed: int = 0,
) -> tuple[DirichletModel, LogisticNormalModel, dict[str, float | bool | int]]:
    """Compare Dirichlet adequacy with a logistic-normal alternative."""

    fit_matrix = close_compositions(fit_values, pseudocount)
    validation_matrix = close_compositions(validation_values, pseudocount)
    dirichlet = fit_dirichlet(fit_matrix, pseudocount=0.0)
    logistic_normal = fit_logistic_normal(fit_matrix, pseudocount=0.0)

    dirichlet_log_likelihood = dirichlet_logpdf(
        validation_matrix, dirichlet.alpha, pseudocount=0.0
    )
    logistic_log_likelihood = logistic_normal_logpdf(
        validation_matrix, logistic_normal, pseudocount=0.0
    )

    empirical_mean = validation_matrix.mean(axis=0)
    empirical_covariance = np.atleast_2d(
        np.cov(validation_matrix, rowvar=False, ddof=1)
    )
    model_mean, model_covariance = dirichlet_moments(dirichlet.alpha)
    positive_fraction, positive_mass_ratio = _positive_covariance_statistics(
        empirical_covariance
    )

    generator = np.random.default_rng(seed)
    simulated = generator.dirichlet(dirichlet.alpha, size=simulation_rows)
    simulated_nll = np.sort(
        -dirichlet_logpdf(simulated, dirichlet.alpha, pseudocount=0.0)
    )
    observed_nll = -dirichlet_log_likelihood
    cdf = np.searchsorted(simulated_nll, observed_nll, side="right") / len(
        simulated_nll
    )
    tail_probability = 1.0 - cdf
    ks = kstest(cdf, "uniform")

    dimensions = validation_matrix.shape[1]
    logistic_dimensions = dimensions - 1
    dirichlet_parameters = dimensions
    logistic_parameters = (
        logistic_dimensions
        + logistic_dimensions * (logistic_dimensions + 1) // 2
    )
    rows = len(validation_matrix)
    dirichlet_aic_per_row = (
        -2.0 * dirichlet_log_likelihood.sum() + 2.0 * dirichlet_parameters
    ) / rows
    logistic_aic_per_row = (
        -2.0 * logistic_log_likelihood.sum() + 2.0 * logistic_parameters
    ) / rows

    covariance_error = np.linalg.norm(
        empirical_covariance - model_covariance, ord="fro"
    ) / max(np.linalg.norm(empirical_covariance, ord="fro"), 1e-12)

    metrics: dict[str, float | bool | int] = {
        "fit_rows": int(len(fit_matrix)),
        "validation_rows": int(len(validation_matrix)),
        "components": int(dimensions),
        "dirichlet_converged": bool(dirichlet.converged),
        "dirichlet_iterations": int(dirichlet.iterations),
        "dirichlet_concentration": float(dirichlet.alpha.sum()),
        "dirichlet_average_log_likelihood": float(
            dirichlet_log_likelihood.mean()
        ),
        "logistic_normal_average_log_likelihood": float(
            logistic_log_likelihood.mean()
        ),
        "dirichlet_minus_logistic_normal_nats": float(
            (dirichlet_log_likelihood - logistic_log_likelihood).mean()
        ),
        "dirichlet_aic_per_row": float(dirichlet_aic_per_row),
        "logistic_normal_aic_per_row": float(logistic_aic_per_row),
        "mean_l1_error": float(np.abs(empirical_mean - model_mean).sum()),
        "covariance_relative_frobenius_error": float(covariance_error),
        "positive_offdiagonal_covariance_fraction": positive_fraction,
        "positive_offdiagonal_covariance_mass_ratio": positive_mass_ratio,
        "moment_concentration_coefficient_of_variation": _moment_concentration_dispersion(
            validation_matrix
        ),
        "nll_pit_ks_statistic": float(ks.statistic),
        "nll_pit_ks_pvalue": float(ks.pvalue),
        "nll_tail_probability_below_001_fraction": float(
            np.mean(tail_probability < 0.01)
        ),
        "nll_tail_probability_below_005_fraction": float(
            np.mean(tail_probability < 0.05)
        ),
    }
    return dirichlet, logistic_normal, metrics
