import numpy as np

from experiments.grounded_route.graph_effectiveness.metrics import (
    binary_metrics,
    paired_source_delta,
)


def test_binary_metrics_include_prevalence_and_lift():
    label = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    result = binary_metrics(label, score)

    assert result["auroc"] == 1.0
    assert result["auprc"] == 1.0
    assert result["prevalence"] == 0.5
    assert result["auprc_lift"] == 2.0


def test_paired_identical_scores_have_zero_delta():
    label = np.asarray([0, 1, 0, 1, 0, 1])
    score = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    source = np.asarray(["a", "a", "b", "b", "c", "c"])
    result = paired_source_delta(
        label,
        score,
        score,
        source,
        replicates=20,
        seed=3,
    )

    assert result["auroc_delta"] == 0.0
    assert result["auprc_delta"] == 0.0
    assert result["auroc_delta_ci_low"] == 0.0
    assert result["auprc_delta_ci_high"] == 0.0
