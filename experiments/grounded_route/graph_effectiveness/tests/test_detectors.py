import numpy as np

from experiments.grounded_route.graph_effectiveness.detectors import (
    DETECTOR_NAMES,
    DetectorConfig,
    score_detectors,
)


def test_all_unsupervised_detectors_score_only_node_matrices():
    random = np.random.default_rng(7)
    calibration = random.normal(size=(80, 8)).astype(np.float32)
    test = random.normal(size=(20, 8)).astype(np.float32)
    scores = score_detectors(
        calibration,
        test,
        config=DetectorConfig(
            components=4,
            neighbors=5,
            max_reference=80,
            one_class_max_reference=80,
            isolation_trees=10,
            neural_hidden_dim=12,
            neural_latent_dim=4,
            neural_epochs=1,
            batch_size=16,
            seed=11,
            neural_seeds=(11,),
        ),
        device="cpu",
    )

    assert set(scores) == set(DETECTOR_NAMES)
    assert all(value.shape == (20,) for value in scores.values())
    assert all(np.isfinite(value).all() for value in scores.values())
