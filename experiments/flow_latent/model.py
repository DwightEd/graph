"""Conditional mixture PPCA: analytic E-step, ridge/spectral M-step."""
import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def fit_projection(values, dimensions):
    projection = PCA(n_components=dimensions, svd_solver='randomized',
                     random_state=42).fit(values)
    return dict(mean=projection.mean_, axes=projection.components_,
                scale=np.sqrt(projection.explained_variance_).clip(.01),
                variance_retained=float(projection.explained_variance_ratio_.sum()))


def project(values, projection):
    return ((values - projection['mean']) @ projection['axes'].T) / projection['scale']


def component_log_density(nodes, context, coefficients, covariances):
    augmented = np.column_stack((np.ones(len(nodes)), context))
    densities = []
    for coefficient, covariance in zip(coefficients, covariances):
        difference = nodes - augmented @ coefficient
        factor = np.linalg.cholesky(covariance)
        standardized = np.linalg.solve(factor, difference.T).T
        log_determinant = 2 * np.log(np.diag(factor)).sum()
        densities.append(-.5 * (np.square(standardized).sum(1) + log_determinant
                               + nodes.shape[1] * np.log(2 * np.pi)))
    return np.column_stack(densities)


def conditional_log_density(nodes, context, model):
    densities = component_log_density(nodes, context, model['coefficients'], model['covariances'])
    weighted = densities + np.log(model['weights'])
    marginal = logsumexp(weighted, axis=1)
    posterior = np.exp(weighted - marginal[:, None])
    return marginal, posterior


def update_components(nodes, context, posterior, rank=4):
    augmented = np.column_stack((np.ones(len(nodes)), context))
    coefficients, covariances, noise, loadings = [], [], [], []
    # Matrix-normal prior on non-intercept coefficients, conditional on covariance.
    penalty = np.eye(augmented.shape[1]) * (.01 * len(nodes))
    penalty[0, 0] = 0
    for probability in posterior.T:
        count = probability.sum()
        gram = augmented.T @ (probability[:, None] * augmented)
        coefficient = np.linalg.solve(gram + penalty,
            augmented.T @ (probability[:, None] * nodes))
        residual = nodes - augmented @ coefficient
        scatter = residual.T @ (probability[:, None] * residual)
        scatter += coefficient.T @ penalty @ coefficient
        covariance = scatter / (count + context.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        variance = max(float(eigenvalues[:-rank].mean()), .05)
        directions = eigenvectors[:, -rank:]
        strengths = np.maximum(eigenvalues[-rank:] - variance, 0)
        loading = directions * np.sqrt(strengths)
        covariance = loading @ loading.T + variance * np.eye(nodes.shape[1])
        coefficients.append(coefficient)
        covariances.append(covariance)
        noise.append(variance)
        loadings.append(loading)
    return np.stack(coefficients), np.stack(covariances), noise, np.stack(loadings)


def map_objective(density, context, model):
    objective = float(density.sum() + .2 * np.log(model['weights']).sum())
    penalty_strength = .01 * len(density)
    for coefficient, covariance in zip(model['coefficients'], model['covariances']):
        coefficient = coefficient[1:]
        log_determinant = np.linalg.slogdet(covariance)[1]
        penalty = penalty_strength * coefficient.T @ coefficient
        objective -= .5 * (context.shape[1] * log_determinant
                           + np.trace(np.linalg.solve(covariance, penalty)))
    return objective


def factor_posterior(nodes, context, model):
    augmented = np.column_stack((np.ones(len(nodes)), context))
    means, covariances = [], []
    for coefficient, covariance, loading in zip(
            model['coefficients'], model['covariances'], model['loadings']):
        residual = nodes - augmented @ coefficient
        precision_loading = np.linalg.solve(covariance, loading)
        means.append(residual @ precision_loading)
        covariances.append(np.eye(loading.shape[1]) - loading.T @ precision_loading)
    return np.stack(means, axis=1), np.stack(covariances)


def fit_mixture(nodes, context, seed=42, iterations=30, rank=4):
    labels = KMeans(n_clusters=4, n_init=1, random_state=seed).fit_predict(nodes)
    posterior = np.full((len(nodes), 4), .01 / 4)
    posterior[np.arange(len(nodes)), labels] += .99
    trace = []
    for iteration in range(iterations):
        coefficients, covariances, noise, loadings = update_components(nodes, context, posterior, rank)
        counts = posterior.sum(0)
        weights = (counts + .2) / (counts.sum() + .8)
        model = dict(coefficients=coefficients, covariances=covariances, weights=weights,
                     loadings=loadings, noise=np.asarray(noise))
        density, posterior = conditional_log_density(nodes, context, model)
        objective = map_objective(density, context, model)
        if trace:
            assert objective >= trace[-1]['map_objective'] - 1e-7 * len(nodes), 'MAP EM objective decreased'
        trace.append(dict(iteration=iteration, mean_log_density=float(density.mean()),
                          map_objective=objective,
                          weights=weights.tolist(), noise=noise,
                          min_effective_tokens=float(posterior.sum(0).min())))
    model['trace'] = trace
    return model
