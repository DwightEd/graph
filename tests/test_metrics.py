import numpy as np
import pytest

from route_graph.metrics import binary_detection_metrics


def test_source_balancing_prevents_long_sources_from_dominating() -> None:
    labels = np.array([1, 1, 1, 1, 1, 0])
    scores = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5])
    sources = np.array(["long"] * 4 + ["short-positive", "short-negative"])

    token_weighted = binary_detection_metrics(
        labels, scores, sources, bootstrap=0, seed=1
    )
    source_balanced = binary_detection_metrics(
        labels, scores, sources, bootstrap=0, seed=1, source_balanced=True
    )

    assert token_weighted["auroc"] == pytest.approx(0.2)
    assert source_balanced["auroc"] == pytest.approx(0.5)
