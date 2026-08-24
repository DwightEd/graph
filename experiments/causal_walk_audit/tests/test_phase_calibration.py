import numpy as np
import torch

from experiments.causal_walk_audit.automaton import P0, R_NEAR
from experiments.causal_walk_audit.calibration import HierarchicalCalibration
from experiments.causal_walk_audit.config import PhaseConfig
from experiments.causal_walk_audit.phase import ChannelStats, score_phase


def test_rupture_is_primary_and_closure_is_separate():
    route = torch.zeros(8, 2, 7)
    route[:4, :, P0] = 1.0
    route[4:, :, R_NEAR] = 1.0
    predicted = route.clone()
    surprisal = torch.zeros(8, 2)
    surprisal[4] = 4.0
    result = score_phase(
        route,
        predicted,
        surprisal,
        stats=ChannelStats(torch.zeros(2), torch.ones(2)),
        config=PhaseConfig(cusum_slack=0.5),
    )
    assert float(result.rupture[4].mean()) > 0
    assert float(result.closure_score[4:].mean()) > 0
    torch.testing.assert_close(
        result.rupture_closure,
        result.rupture * result.closure_score,
    )


def test_hierarchical_calibration_returns_finite_score():
    rng = np.random.default_rng(3)
    channel = rng.normal(size=(200, 4)).astype(np.float32)
    fusion = rng.normal(size=(100, 4)).astype(np.float32)
    calibration = HierarchicalCalibration.fit(
        channel,
        fusion,
        num_layers=2,
        num_heads=2,
    )
    score, layer, probability = calibration.score(
        rng.normal(size=(5, 4)).astype(np.float32)
    )
    assert score.shape == (5,)
    assert layer.shape == (5, 2)
    assert probability.shape == (5, 4)
    assert np.isfinite(score).all()
