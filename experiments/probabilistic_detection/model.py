"""Separate source-conditioned class odds from residual observation evidence."""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler


COVARIANCE_FLOOR = 1e-6


def gaussian_log_density(observations, mean, cholesky):
    """Gaussian log density with rows as observations and a shared covariance."""
    residual = observations - mean
    whitened = solve_triangular(cholesky, residual.T, lower=True, check_finite=False)
    log_determinant = 2 * np.log(np.diag(cholesky)).sum()
    constant = observations.shape[1] * np.log(2 * np.pi)
    return -.5 * (np.square(whitened).sum(axis=0) + log_determinant + constant)


def conditional_mean(design, observations, weights, ridge):
    """Weighted ridge regression; the leading constant receives no penalty."""
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    weighted_design = design * weights[:, None]
    coefficients = np.linalg.solve(
        design.T @ weighted_design + penalty,
        weighted_design.T @ observations,
    )
    residual = observations - design @ coefficients
    covariance = (residual.T * weights) @ residual / weights.sum()
    return coefficients, covariance


def regularized_covariance(covariance, shrinkage, diagonal):
    """Keep class variance; control dependence and numerical singularity separately."""
    variances = np.diag(np.diag(covariance))
    if diagonal:
        covariance = variances
    else:
        covariance = (1 - shrinkage) * covariance + shrinkage * variances
    floor = COVARIANCE_FLOOR * max(float(np.trace(covariance) / len(covariance)), 1.)
    return covariance + floor * np.eye(len(covariance))


def shared_correlation_covariances(covariances, masses, shrinkage):
    """Pool standardized residual dependence while retaining class-specific variance.

    The variance floor matches the existing covariance controls. Numerical jitter
    enters the common correlation before reconstruction, so both final classes
    have exactly the same correlation, including when some channels are constant.
    """
    covariances = np.asarray(covariances)
    variances = np.diagonal(covariances, axis1=1, axis2=2).copy()
    floor = COVARIANCE_FLOOR * np.maximum(variances.mean(axis=1), 1.)
    scale = np.sqrt(np.maximum(variances, floor[:, None]))
    correlations = covariances / (scale[:, :, None] * scale[:, None, :])
    diagonal = np.arange(variances.shape[1])
    correlations[:, diagonal, diagonal] = 1.

    pooled = np.average(correlations, axis=0, weights=masses)
    identity = np.eye(len(pooled))
    pooled = (1 - shrinkage) * pooled + shrinkage * identity
    pooled = (pooled + COVARIANCE_FLOOR * identity) / (1 + COVARIANCE_FLOOR)
    scale = np.sqrt(variances + floor[:, None])
    return pooled[None, :, :] * scale[:, :, None] * scale[:, None, :]


def training_inputs(context, observations, labels, sample_weight):
    """Validate scientific input at the fit boundary and fix the regularization scale."""
    context = np.asarray(context, dtype=float)
    observations = np.asarray(observations, dtype=float)
    labels = np.asarray(labels)
    weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if context.ndim != 2 or observations.ndim != 2:
        raise ValueError("Context and observations must be two-dimensional matrices")
    if labels.shape != (len(context),) or weights.shape != labels.shape or len(observations) != len(labels):
        raise ValueError("Context, observations, labels and weights must share the row axis")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Training requires both binary classes 0 and 1")
    if not all(np.isfinite(values).all() for values in (context, observations, weights)):
        raise ValueError("Training inputs and weights must be finite")
    if (weights <= 0).any():
        raise ValueError("Training weights must be strictly positive")
    weights = weights * (len(weights) / weights.sum())
    return context, observations, labels.astype(int), weights


class ConditionalGaussian:
    """Weighted class-conditional Gaussian residuals around spline context means.

    The common observation transform is fitted without labels on training rows.
    Densities are defined in that frozen transformed space. A Jacobian would
    cancel for an invertible common transform; empirical ties and tail clipping
    prevent claiming exact preservation of a native-space likelihood ratio.
    Gaussian residuals are a modeling assumption about transformed observations.
    """

    def __init__(self, covariance="full", shrinkage=.5, ridge=1.):
        if covariance not in ("full", "diagonal", "shared", "shared_correlation"):
            raise ValueError("Covariance must be full, diagonal, shared or shared_correlation")
        if not 0 <= shrinkage <= 1 or ridge <= 0:
            raise ValueError("Shrinkage must lie in [0, 1] and ridge must be positive")
        self.covariance = covariance
        self.shrinkage = shrinkage
        self.ridge = ridge

    def _fit_transforms(self, context, observations, weights):
        self.context_scaler_ = StandardScaler().fit(context, sample_weight=weights)
        scaled = self.context_scaler_.transform(context)
        self.context_spline_ = SplineTransformer(n_knots=4, degree=2, include_bias=False).fit(scaled)
        self.observation_transform_ = QuantileTransformer(
            n_quantiles=min(1000, len(observations)),
            output_distribution="normal",
            random_state=42,
        ).fit(observations)
        return self._context_design(context), self.observation_transform_.transform(observations)

    def _context_design(self, context):
        scaled = self.context_scaler_.transform(context)
        spline = self.context_spline_.transform(scaled)
        return np.column_stack((np.ones(len(context)), spline))

    def _fit_conditionals(self, design, observations, labels, weights):
        coefficients, covariances, masses = [], [], []
        for label in (0, 1):
            selected = labels == label
            coefficient, covariance = conditional_mean(
                design[selected], observations[selected], weights[selected], self.ridge
            )
            coefficients.append(coefficient)
            covariances.append(covariance)
            masses.append(weights[selected].sum())
        if self.covariance == "shared":
            pooled = np.average(covariances, axis=0, weights=masses)
            covariances = [pooled, pooled]
        self.mean_coefficients_ = np.stack(coefficients)
        if self.covariance == "shared_correlation":
            self.covariances_ = shared_correlation_covariances(covariances, masses, self.shrinkage)
        else:
            self.covariances_ = np.stack([
                regularized_covariance(matrix, self.shrinkage, self.covariance == "diagonal")
                for matrix in covariances
            ])
        self.cholesky_ = np.linalg.cholesky(self.covariances_)

    def fit(self, context, observations, labels, sample_weight=None):
        context, observations, labels, weights = training_inputs(context, observations, labels, sample_weight)
        design, transformed = self._fit_transforms(context, observations, weights)
        self.anchor_ = LogisticRegression(C=1., max_iter=1000, random_state=42)
        self.anchor_.fit(design[:, 1:], labels, sample_weight=weights)
        self._fit_conditionals(design, transformed, labels, weights)
        self.n_context_features_in_ = context.shape[1]
        self.n_observation_features_in_ = observations.shape[1]
        return self

    def predict_components(self, context, observations):
        """Score new rows using frozen transforms, context odds and class residuals."""
        context = np.asarray(context, dtype=float)
        observations = np.asarray(observations, dtype=float)
        design = self._context_design(context)
        transformed = self.observation_transform_.transform(observations)
        anchor = self.anchor_.decision_function(design[:, 1:])
        densities = [
            gaussian_log_density(transformed, design @ self.mean_coefficients_[label], self.cholesky_[label])
            for label in (0, 1)
        ]
        ratio = densities[1] - densities[0]
        return dict(anchor_logit=anchor, conditional_log_ratio=ratio, score=anchor + ratio)
