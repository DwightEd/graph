import numpy as np

from experiments.grounded_route.evaluation.detectors import (
    DetectorConfig,
    score_detectors,
    score_pca_knn,
)


def test_pca_knn_assigns_larger_score_to_far_node():
    random = np.random.default_rng(7)
    calibration = random.normal(size=(200, 8)).astype(np.float32)
    test = np.vstack((np.zeros((1, 8)), np.full((1, 8), 8.0))).astype(np.float32)
    score = score_pca_knn(
        calibration,
        test,
        DetectorConfig(components=4, neighbors=10, seeds=(7,)),
    )
    assert score[1] > score[0]


def test_collapsed_embeddings_return_trivial_scores():
    calibration = np.ones((40, 8), dtype=np.float32)
    test = np.ones((5, 8), dtype=np.float32)
    scores = score_detectors(
        calibration,
        test,
        DetectorConfig(components=4, neighbors=5, epochs=1, seeds=(7,)),
    )
    assert all(np.array_equal(score, np.zeros(5)) for score in scores.values())
