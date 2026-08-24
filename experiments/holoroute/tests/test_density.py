import numpy as np
from scipy.stats import spearmanr

from experiments.holoroute.density import ConditionalDensity


def test_conditional_density_removes_position_trend():
    rng = np.random.default_rng(4)
    rows = 400
    position = np.linspace(0, 1, rows)
    nuisance = np.column_stack((position, position**2, position**3))
    feature = np.column_stack(
        (
            4.0 * position + rng.normal(0, 0.1, rows),
            2.0 * position**2 + rng.normal(0, 0.1, rows),
        )
    )
    task = np.repeat("QA", rows)
    density = ConditionalDensity.fit(
        feature,
        nuisance,
        task,
        ridge_alpha=1e-6,
        covariance_shrinkage=0.2,
        scale_floor=1e-3,
    )
    score, standardized = density.score(feature, nuisance, task)
    assert abs(spearmanr(score, position).statistic) < 0.15
    assert abs(spearmanr(standardized[:, 0], position).statistic) < 0.15


def test_density_ignores_unavailable_mechanism_column():
    feature = np.column_stack((np.linspace(0, 1, 20), np.full(20, np.nan)))
    nuisance = np.column_stack((np.linspace(0, 1, 20), np.ones(20)))
    task = np.repeat("QA", 20)
    density = ConditionalDensity.fit(
        feature,
        nuisance,
        task,
        ridge_alpha=1e-3,
        covariance_shrinkage=0.2,
        scale_floor=1e-3,
    )
    score, standardized = density.score(feature, nuisance, task)
    assert not density.active_feature[1]
    assert np.all(standardized[:, 1] == 0)
    assert np.isfinite(score).all()
