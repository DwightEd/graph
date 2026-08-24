import numpy as np
import torch

from experiments.causal_walk_audit.lineage import LineageTrace
from experiments.causal_walk_audit.trajectory import (
    TrajectoryReference,
    layer_trajectory,
    summarize_trajectory,
)


def _trace(mismatch: bool):
    state = torch.zeros(3, 3, 2, 3, 3)
    state[..., 2, 0] = 0.2
    state[..., 0, 0] = 0.4
    state[..., 0 if not mismatch else 1, 1] = 0.4
    unresolved = torch.zeros(3, 3, 2)
    return LineageTrace(
        state=state,
        unresolved=unresolved,
        anchor_count=2,
        anchor_mode="test",
    )


def test_anchor_js_and_lock_in_are_directional():
    aligned = layer_trajectory(_trace(False), minimum_anchor_mass=1e-4)
    fractured = layer_trajectory(_trace(True), minimum_anchor_mass=1e-4)
    assert float(np.nanmean(fractured.anchor_js)) > float(np.nanmean(aligned.anchor_js))

    reference = TrajectoryReference(
        js_high=0.2,
        js_low=0.05,
        evidence_high=0.7,
        evidence_low=0.3,
        response_high=0.1,
    )
    summary = summarize_trajectory(fractured, reference, horizon=2)
    assert summary.anchor_js_peak.mean() > 0
    assert np.all(summary.lock_in >= 0)
