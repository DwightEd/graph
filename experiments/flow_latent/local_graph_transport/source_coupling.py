"""FIT-only Gaussian source-pair/route dependence ratio in normal coordinates.

The ratio describes statistical coupling, not factual correctness. It preserves
the source pair and subtracts marginal rarity; its residual sign symmetry is an
explicit limitation. No entropy, source entity annotations or token averaging.
"""
import numpy as np
from scipy.special import ndtri

from .unlabeled import equal_source_weights, fit_weighted_cdf, reference_rank


COORDINATES = ('source_local', 'source_full', 'raw_route',
               'local_native_logp', 'full_native_logp')
SHRINKAGE = 1 / 672


def observable_endpoints(reference, name):
    cumulative = np.asarray(reference[name + '_cumulative'], dtype=np.float64)
    if (len(cumulative) < 2 or cumulative[0] != 0 or not np.isfinite(cumulative).all()
            or not np.all(np.diff(cumulative) > 0)):
        raise ValueError('A normalized empirical CDF with observable support is required')
    lower, upper = cumulative[1] / 2, (cumulative[-2] + 1) / 2
    if not 0 < lower <= upper < 1:
        raise ValueError('Observable midCDF endpoints must be strictly inside (0,1)')
    return lower, upper


def normal_score(ranks, reference, name):
    lower, upper = observable_endpoints(reference, name)
    ranks = np.asarray(ranks, dtype=np.float64)
    if not np.isfinite(ranks).all():
        raise ValueError('Finite empirical ranks are required')
    return ndtri(np.clip(ranks, lower, upper))


def fit_normal_model(fit_ranks, source_index, global_reference):
    """Fit exactly five coordinates using the complete valid FIT token reference."""
    source_index = np.asarray(source_index)
    weights = equal_source_weights(source_index)
    reference, normals, diagnostics = {}, [], {}
    for name in COORDINATES:
        values = np.asarray(fit_ranks[name], dtype=np.float64)
        if values.shape != source_index.shape or not np.isfinite(values).all():
            raise ValueError('All FIT coordinates must have one complete finite token axis')
        if name in COORDINATES[:3]:
            ranks, cdf = values, global_reference
        else:
            distinct, cumulative = fit_weighted_cdf(values, weights)
            reference[name + '_values'] = distinct
            reference[name + '_cumulative'] = cumulative
            cdf = reference
            ranks = reference_rank(cdf, name, values)
        normal = normal_score(ranks, cdf, name)
        normals.append(normal)
        diagnostics[name + '_rank'] = ranks
        diagnostics[name + '_normal'] = normal
    z = np.stack(normals, axis=1)
    mean = weights @ z
    second = z.T @ (z * weights[:, None])
    covariance = second - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    if np.any(np.diag(covariance) <= 0):
        raise ValueError('Constant Gaussian coordinates fail the frozen model')
    shrunk = ((1 - SHRINKAGE) * covariance
              + SHRINKAGE * np.diag(np.diag(covariance)))
    conditioner = np.linalg.solve(shrunk[3:, 3:], shrunk[3:, :3]).T
    conditional = shrunk[:3, :3] - conditioner @ shrunk[3:, :3]
    conditional = (conditional + conditional.T) / 2
    A, b, d = conditional[:2, :2], conditional[:2, 2], float(conditional[2, 2])
    beta = np.linalg.solve(A, b)
    v = float(d - b @ beta)
    if (not np.isfinite(shrunk).all() or np.linalg.eigvalsh(shrunk).min() <= 0
            or np.linalg.eigvalsh(conditional).min() <= 0 or not 0 < v <= d):
        raise ValueError('Fixed Gaussian conditional model is not finite positive definite')
    model = dict(mean=mean, raw_covariance=covariance, covariance=shrunk,
                 conditioner=conditioner, conditional_covariance=conditional,
                 A=A, b=b, d=np.asarray(d), beta=beta, v=np.asarray(v),
                 shrinkage=np.asarray(SHRINKAGE))
    return model, reference, z, diagnostics


def dependence_ratio(normal_coordinates, model):
    """Exact log(product-of-block-marginals/joint) for the frozen Gaussian."""
    z = np.asarray(normal_coordinates, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 5 or not np.isfinite(z).all():
        raise ValueError('Normal coordinates must be finite [token,5]')
    mean = np.asarray(model['mean'])
    residual = z[:, :3] - mean[:3] - (z[:, 3:] - mean[3:]) @ model['conditioner'].T
    predicted = residual[:, :2] @ model['beta']
    route_error = residual[:, 2] - predicted
    d, v = float(model['d']), float(model['v'])
    if not 0 < v <= d:
        raise ValueError('Positive conditional/marginal route variance is required')
    D = .5 * np.log(v / d) + .5 * (route_error**2 / v - residual[:, 2]**2 / d)
    if not np.isfinite(D).all():
        raise ValueError('Every token requires a finite dependence ratio')
    diagnostics = dict(D=D, source_local_residual=residual[:, 0],
        source_full_residual=residual[:, 1], route_residual=residual[:, 2],
        source_predicted_route=predicted, conditional_route_error=route_error)
    return D, diagnostics


def answer_coordinates(global_reference, native_reference, values, likelihood):
    """Read every answer node; valid-token filtering happens after graph inference."""
    normals, diagnostics = [], {}
    for name in COORDINATES:
        if name in COORDINATES[:3]:
            ranks = reference_rank(global_reference, name, values[name])
            cdf = global_reference
        else:
            view = 'local' if name.startswith('local') else 'full'
            ranks = reference_rank(native_reference, name, likelihood[view])
            cdf = native_reference
        normal = normal_score(ranks, cdf, name)
        normals.append(normal)
        diagnostics[name + '_rank'] = ranks
        diagnostics[name + '_normal'] = normal
    return np.stack(normals, axis=1), diagnostics


def fit_risk_reference(D, source_index):
    values, cumulative = fit_weighted_cdf(D, equal_source_weights(source_index))
    return dict(coupling_values=values, coupling_cumulative=cumulative)


def coupling_rank(reference, D):
    return reference_rank(reference, 'coupling', np.asarray(D, dtype=np.float64))
