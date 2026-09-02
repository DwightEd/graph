from types import SimpleNamespace

import numpy as np
import torch

from experiments.evidence_route_state.detector import StickyRouteHMM
from experiments.evidence_route_state.state import (
    EquationLockedRouteCollapseControl,
    build_route_state,
    prompt_log_volume,
    route_observation,
    route_volume,
)


def lineage_with_history(*, unrooted: bool) -> SimpleNamespace:
    """The same narrow topology, rooted either in evidence or response history."""

    layers, tokens, heads = 1, 3, 2

    direct = torch.zeros(layers, tokens, heads)
    relay = torch.zeros_like(direct)
    feedback = torch.zeros_like(direct)
    if unrooted:
        feedback[:, 2] = 1.0
    else:
        relay[:, 2] = 1.0

    return SimpleNamespace(
        query_position=torch.tensor([2, 3, 4]),
        prediction_position=torch.tensor([3, 4, 5]),
        history_valid=torch.tensor([False, False, True]),
        prompt_evidence=direct,
        grounded_response_relay=relay,
        unrooted_response_feedback=feedback,
        effective_sources=torch.ones(layers, tokens),
        effective_head_rank=torch.ones(layers, tokens),
        anchor_source=torch.zeros(layers, tokens, heads, dtype=torch.long),
    )


def route_state_hmm() -> StickyRouteHMM:
    model = StickyRouteHMM()
    model.initial_ = np.full(3, 1.0 / 3.0)
    model.transition_ = np.array(
        [
            [0.9, 0.09, 0.01],
            [0.05, 0.9, 0.05],
            [0.01, 0.09, 0.9],
        ]
    )
    model.means_ = np.array([[0.1, 0.2], [0.9, 0.0], [0.9, 1.0]])
    model.variances_ = np.full((3, 2), 0.01)
    return model


def test_narrow_evidence_rooted_focus_is_not_captured():
    grounded = build_route_state(lineage_with_history(unrooted=False))
    captured = build_route_state(lineage_with_history(unrooted=True))

    # The old route-collapse observation cannot distinguish these cases.
    np.testing.assert_allclose(grounded.raw_contraction, captured.raw_contraction)
    assert grounded.raw_contraction[2] > 0.9

    assert grounded.takeover[2] == 0.0
    assert captured.takeover[2] == 1.0

    model = route_state_hmm()
    grounded_observation = route_observation(
        grounded.raw_contraction, grounded.takeover, grounded.valid
    )
    captured_observation = route_observation(
        captured.raw_contraction, captured.takeover, captured.valid
    )
    grounded_score = model.score(grounded_observation, grounded.valid)[2]
    captured_score = model.score(captured_observation, captured.valid)[2]
    assert grounded_score < 0.05
    assert captured_score > 0.95


def test_route_volume_uses_compact_source_token_topology():
    volume = route_volume(
        effective_sources=np.array([[2.0, 4.0]]),
        effective_head_rank=np.array([[2.0, 1.0]]),
        anchor_source=np.array([[[0, 0], [0, 1]]]),
        query_position=np.array([3, 4]),
        anchor_window=1,
    )

    np.testing.assert_allclose(
        volume.log_volume,
        np.array([[np.log(4.0), np.log(8.0)]]),
    )
    assert np.all((volume.normalized >= 0.0) & (volume.normalized <= 1.0))


def test_locked_prompt_volume_is_the_sum_of_three_log_degrees():
    effective_sources = np.array([[2.0, 4.0]])
    effective_rank = np.array([[2.0, 1.0]])
    anchor_source = np.array([[[0, 0], [0, 1]]])

    volume = prompt_log_volume(
        effective_sources,
        effective_rank,
        anchor_source,
        window=1,
    )

    np.testing.assert_allclose(volume, np.array([[np.log(4.0)], [np.log(8.0)]]))


def collapse_records(label: int):
    rng = np.random.default_rng(19)
    records = []
    for source in range(8):
        tokens = 20
        prompt_length = 80 + 10 * source
        position = (np.arange(tokens) + 0.5) / tokens
        base = 2.0 + 0.4 * position + 0.15 * position**2
        base += 0.08 * np.log1p(prompt_length + tokens)
        volume = np.column_stack(
            (
                base + rng.normal(scale=0.08, size=tokens),
                0.7 * base + rng.normal(scale=0.08, size=tokens),
            )
        )
        records.append(
            {
                "source_id": f"source-{source}",
                "prompt_length": prompt_length,
                "volume": volume,
                "label": label,
            }
        )
    return records


def test_locked_collapse_control_is_label_invariant_and_scores_narrow_routes_high():
    correct_labels = collapse_records(label=0)
    flipped_labels = collapse_records(label=1)
    control = EquationLockedRouteCollapseControl.fit(
        correct_labels[:6], correct_labels[6:]
    )
    flipped = EquationLockedRouteCollapseControl.fit(
        flipped_labels[:6], flipped_labels[6:]
    )

    np.testing.assert_array_equal(control.coefficients, flipped.coefficients)
    np.testing.assert_array_equal(control.scales, flipped.scales)
    for left, right in zip(
        control.position_tables, flipped.position_tables, strict=True
    ):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])

    reference = correct_labels[0]
    narrow = {**reference, "volume": reference["volume"] - 1.0, "label": 1}
    wide = {**reference, "volume": reference["volume"] + 1.0, "label": 0}
    assert control.score(narrow).mean() > control.score(wide).mean()


def test_locked_collapse_control_round_trips_all_calibration_state(tmp_path):
    records = collapse_records(label=0)
    control = EquationLockedRouteCollapseControl.fit(records[:6], records[6:])
    path = tmp_path / "collapse.npz"

    control.save(path)
    restored = EquationLockedRouteCollapseControl.load(path)

    np.testing.assert_array_equal(restored.coefficients, control.coefficients)
    np.testing.assert_array_equal(restored.scales, control.scales)
    np.testing.assert_array_equal(restored.score(records[0]), control.score(records[0]))
