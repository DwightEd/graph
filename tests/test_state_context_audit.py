import numpy as np
import pandas as pd

from experiments.charm_structure_audit.state_context_audit import temporal_views, safe_metrics


def test_temporal_views_never_cross_answer_boundaries():
    table = pd.DataFrame(dict(
        id=["a", "a", "a", "b", "b"],
        previous_gold=[-1, 0, 1, -1, 0],
    ))
    values = np.arange(10, dtype=float).reshape(5, 2)
    previous, past = temporal_views(values, table, window=2)

    assert np.isnan(previous[0]).all()
    np.testing.assert_array_equal(previous[1], values[0])
    np.testing.assert_array_equal(previous[2], values[1])
    assert np.isnan(previous[3]).all()
    np.testing.assert_array_equal(previous[4], values[3])
    np.testing.assert_allclose(past[2], values[:2].mean(axis=0))


def test_safe_metrics_reports_unidentifiable_fixed_stratum():
    result = safe_metrics(np.array([1, 1]), np.array([0.2, 0.3]))
    assert np.isnan(result["auroc"])
    assert result["positives"] == 2
