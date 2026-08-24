import numpy as np
from scipy.stats import spearmanr

from experiments.attention_holonomy_audit.config import ReferenceConfig
from experiments.attention_holonomy_audit.reference import fit_nuisance_reference


def test_position_residualizer_removes_polynomial_position_trend():
    rng = np.random.default_rng(3)
    rows = 400
    position = np.linspace(0, 1, rows)
    nuisance = np.stack(
        (
            np.arange(rows),
            position,
            np.full(rows, rows),
            np.full(rows, 5),
            np.full(rows, 4),
            np.full(rows, 2),
            np.full(rows, 0.5),
            np.full(rows, 0.8),
            np.full(rows, 0.1),
        ),
        axis=1,
    )
    trend = 2 * position + 3 * position**2 - position**3
    primary = np.stack(
        [trend + 0.05 * rng.normal(size=rows) for _ in range(6)], axis=1
    )
    task = np.repeat("QA", rows)
    reference = fit_nuisance_reference(
        primary,
        nuisance,
        task,
        config=ReferenceConfig(position_degree=3),
    )
    standardized, _ = reference.transform(primary, nuisance, task)
    correlation = spearmanr(standardized[:, 0], position).statistic
    assert abs(correlation) < 0.05
