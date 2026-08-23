import numpy as np

from experiments.non_neural_structure_audit.statistics import (
    benjamini_hochberg,
    grouped_bootstrap_delta,
    grouped_metric_interval,
)


def test_benjamini_hochberg_is_monotone_in_ranked_p_values():
    adjusted = benjamini_hochberg(np.asarray([0.01, 0.04, 0.03]))

    np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])


def test_group_bootstrap_resamples_whole_samples():
    labels = [np.asarray([0, 1]), np.asarray([0, 1])]
    real = [np.asarray([0.1, 0.9]), np.asarray([0.2, 0.8])]
    baseline = [np.asarray([0.9, 0.1]), np.asarray([0.8, 0.2])]

    result = grouped_bootstrap_delta(
        labels,
        real,
        baseline,
        replicates=20,
        seed=7,
    )

    assert result["auprc_delta"] > 0
    assert result["auprc_delta_ci_low"] > 0


def test_single_class_smoke_returns_undefined_metrics_instead_of_crashing():
    result = grouped_metric_interval(
        [np.zeros(4, dtype=np.int8)],
        [np.arange(4, dtype=np.float32)],
        replicates=5,
        seed=2,
    )

    assert np.isnan(result["auprc"])
    assert np.isnan(result["auprc_ci_low"])
