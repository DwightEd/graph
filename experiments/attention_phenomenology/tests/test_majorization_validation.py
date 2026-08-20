from types import SimpleNamespace

import numpy as np

from experiments.attention_phenomenology.majorization_validation import (
    evaluate_majorization_rows,
)


def test_evaluation_separates_current_detection_from_next_token_forecast():
    rows = {
        "sample_id": np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"]),
        "token_index": np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
        "valid": np.ones(8, dtype=bool),
        "majorization_evidence": np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=float),
        "concentration_level": np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=float),
        "hill_shape": np.zeros(8),
        "source_affinity": np.asarray([1, 1, 0.1, 0.9, 1, 1, 1, 1], dtype=float),
        "entry_probability": np.asarray([0, 0, 0.9, 0.1, 0, 0, 0, 0], dtype=float),
        "basin_probability": np.asarray([0, 0, 0.1, 0.9, 0, 0, 0, 0], dtype=float),
        "current_probability": np.asarray([0, 0, 1, 1, 0, 0, 0, 0], dtype=float),
        "forecast_probability": np.asarray([0, 1, 1, 0, 0, 0, 0, 0], dtype=float),
        "uniform_current_probability": np.asarray(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float
        ),
    }
    labels = SimpleNamespace(
        token_label=np.asarray([0, 0, 1, 1, 0, 0, 0, 0]),
    )

    report = evaluate_majorization_rows(
        rows,
        labels,
        bootstrap_replicates=10,
        seed=4,
    )

    assert report["current_detection"]["auroc"] == 1.0
    assert report["forecast"]["horizon_1"]["auroc"] == 1.0
    assert report["control_metrics"]["uniform"]["real_minus_control_auroc"] == 0.5
    assert report["tokens"] == 8
    assert report["valid_tokens"] == 8


def test_evaluation_reports_zero_coverage_without_crashing():
    rows = {
        "sample_id": np.asarray(["a", "a"]),
        "token_index": np.asarray([0, 1]),
        "valid": np.zeros(2, dtype=bool),
        **{
            name: np.full(2, np.nan)
            for name in (
                "majorization_evidence",
                "concentration_level",
                "hill_shape",
                "source_affinity",
                "entry_probability",
                "basin_probability",
                "current_probability",
                "forecast_probability",
            )
        },
    }
    labels = SimpleNamespace(token_label=np.asarray([0, 1]))

    report = evaluate_majorization_rows(
        rows,
        labels,
        bootstrap_replicates=2,
        seed=1,
    )

    assert report["valid_tokens"] == 0
    assert np.isnan(report["current_detection"]["auroc"])
