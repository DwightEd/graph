import numpy as np

from experiments.attention_mechanism_audit.aggregation import aggregate_trajectory


def test_nan_aggregation_never_imputes_unavailable_token_as_zero():
    values = np.asarray([999.0, 2.0, 4.0, 1.0])
    available = np.asarray([False, True, True, True])
    result = aggregate_trajectory(values, available)
    np.testing.assert_allclose(result["mean"], 7.0 / 3.0)
    np.testing.assert_allclose(result["early"], 2.0)
    np.testing.assert_allclose(result["late"], 2.5)
    np.testing.assert_allclose(result["late_minus_early"], 0.5)
    np.testing.assert_allclose(result["max"], 4.0)
    np.testing.assert_allclose(result["max_adjacent_drop"], 3.0)
    assert int(result["available_count"]) == 3
    assert int(result["adjacent_available_count"]) == 2


def test_all_unavailable_statistics_are_nan_not_zero():
    result = aggregate_trajectory(
        np.asarray([1.0, 2.0]), np.asarray([False, False])
    )
    for name in (
        "mean",
        "early",
        "late",
        "late_minus_early",
        "max",
        "max_adjacent_drop",
    ):
        assert np.isnan(result[name])
    assert int(result["available_count"]) == 0
