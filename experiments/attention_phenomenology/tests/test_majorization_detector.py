import pytest
import torch

from experiments.attention_phenomenology.majorization_detector import (
    CausalMajorizationDetector,
    MajorizationDetectorConfig,
)
from experiments.attention_phenomenology.majorization_dynamics import CausalRouteTrace


def trace(majorization, concentration, affinity, valid=None):
    majorization = torch.tensor(majorization, dtype=torch.float32)
    concentration = torch.tensor(concentration, dtype=torch.float32)
    affinity = torch.tensor(affinity, dtype=torch.float32)
    if valid is None:
        valid = torch.ones_like(majorization, dtype=torch.bool)
    else:
        valid = torch.tensor(valid, dtype=torch.bool)
    return CausalRouteTrace(
        majorization_evidence=majorization,
        concentration_level=concentration,
        hill_shape=torch.zeros_like(majorization),
        source_affinity=affinity,
        valid_channel_fraction=valid.float(),
        valid=valid,
    )


def test_detector_fits_without_labels_and_scores_every_token():
    training = [
        trace([-0.2, 0.0, 0.2], [-0.3, 0.0, 0.3], [0.8, 0.8, 0.8]),
        trace([-0.1, 0.0, 0.1], [-0.2, 0.0, 0.2], [0.9, 0.9, 0.9]),
    ]
    detector = CausalMajorizationDetector.fit(
        training,
        config=MajorizationDetectorConfig(minimum_scale=0.01),
    )

    scores = detector.score(
        trace([-0.1, 1.0, 1.0], [-0.2, 1.2, 1.2], [0.9, 0.1, 0.95])
    )

    assert len(scores.current_probability) == 3
    assert scores.entry_probability[1] > scores.basin_probability[1]
    assert scores.basin_probability[2] > scores.entry_probability[2]
    assert detector.labels_read is False


def test_invalid_gap_is_reported_and_does_not_carry_a_basin_state():
    detector = CausalMajorizationDetector.fit(
        [trace([-0.1, 0.0, 0.1], [-0.1, 0.0, 0.1], [0.8, 0.8, 0.8])]
    )
    scores = detector.score(
        trace(
            [1.0, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.95, 0.0, 0.9],
            valid=[True, False, True],
        )
    )

    assert torch.isnan(scores.current_probability[1])
    assert scores.state_probability[2, 0] > 0.5


def test_detector_rejects_non_positive_robust_scale_floor():
    with pytest.raises(ValueError, match="minimum_scale"):
        CausalMajorizationDetector.fit(
            [trace([0.0], [0.0], [1.0])],
            config=MajorizationDetectorConfig(minimum_scale=0.0),
        )
