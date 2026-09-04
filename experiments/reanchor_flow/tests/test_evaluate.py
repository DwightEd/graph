import numpy as np

from experiments.reanchor_flow.evaluate import aligned_curves
from experiments.reanchor_flow.metrics import metric, paired_bootstrap


def test_metric_uses_higher_score_as_hallucination_risk():
    result = metric([0, 0, 1, 1], [0.0, 0.2, 0.8, 1.0])
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0


def test_bootstrap_difference_resamples_identical_source_clusters():
    label = np.tile([0, 1], 4)
    first = np.tile([0.0, 1.0], 4)
    second = np.tile([1.0, 0.0], 4)
    source = np.repeat(["a", "b", "c", "d"], 2)
    result = paired_bootstrap(label, first, second, source, repeats=20, seed=3)
    assert result["auroc_difference"] == 1.0
    assert result["auroc_difference_ci95"] == [1.0, 1.0]


def test_claim_aligned_curves_use_absolute_claim_coordinates():
    row = {
        "claim_start": np.array([5]),
        "claim_stop": np.array([8]),
        "claim_label": np.array([True]),
        "response_start": 3,
        "functional_evidence_inflow": np.array([0.1, 0.2, 0.9, 0.4, 0.3]),
        "functional_history_inflow": np.array([0.9, 0.8, 0.1, 0.6, 0.7]),
    }
    curves = aligned_curves([row], low=-1, high=1)
    np.testing.assert_allclose(
        curves["hallucinated_evidence"], [0.2, 0.9, 0.4]
    )
    np.testing.assert_allclose(
        curves["hallucinated_history"], [0.8, 0.1, 0.6]
    )
