from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from experiments.evidence_route_state.detector import (
    HISTORY,
    GraphRecord,
    SourceECDF,
    TransitionDetector,
    candidate_windows,
    prompt_length_edges,
)
from experiments.evidence_route_state.graph import GraphSequence
from experiments.evidence_route_state.metric import BLOCK_NAMES


def graph(
    value: float | np.ndarray, *, valid: np.ndarray | None = None
) -> GraphSequence:
    trajectory = np.atleast_1d(np.asarray(value, dtype=np.float32))
    tokens = len(trajectory)
    layers, heads, channels, hidden = 2, 2, 4, 3

    def filled(shape: tuple[int, ...]) -> torch.Tensor:
        result = np.zeros((tokens, *shape), dtype=np.float32)
        result.reshape(tokens, -1)[:] = trajectory[:, None]
        return torch.from_numpy(result)

    prediction = torch.arange(40, 40 + tokens)
    return GraphSequence(
        query_position=prediction - 1,
        prediction_position=prediction,
        node_embedding=filled((channels, hidden)),
        residual_gram=filled((layers + 1, channels, channels)),
        head_write_gram=filled((layers, heads, channels, channels)),
        route_topology=filled((layers, heads, channels, 7)),
        mlp_relation=filled((layers, channels + 1)),
        margin_contribution=filled((channels,)),
        valid=torch.as_tensor(np.ones(tokens, dtype=bool) if valid is None else valid),
    )


def record(source: str, value: float | np.ndarray, prompt: int = 100) -> GraphRecord:
    return GraphRecord(source, prompt, graph(value))


def references() -> tuple[GraphRecord, ...]:
    """Two common modes: broad exploration and grounded narrow focus."""

    return (
        record("broad-0", np.zeros(12)),
        record("broad-1", np.full(12, 0.1)),
        record("focus-0", np.full(12, 4.0)),
        record("focus-1", np.full(12, 4.1)),
    )


def test_candidates_are_source_balanced_within_each_position_decile():
    records = (
        record("shared", np.zeros(21)),
        record("shared", np.ones(21)),
        record("other", np.full(21, 2.0)),
    )
    candidates = candidate_windows(records, prompt_length_edges(records))
    coordinates = [(item.source_id, item.position_bin) for item in candidates]

    assert len(coordinates) == len(set(coordinates))
    assert all(item.end >= HISTORY for item in candidates)


def test_conditional_energy_uses_order_while_raw_control_uses_only_current_frame():
    detector = TransitionDetector(prototype_count=2).fit(references())
    token = 7
    grounded = record("test-grounded", np.full(12, 4.0))
    switch = np.zeros(12)
    switch[token] = 4.0
    switched = record("test-switch", switch)

    grounded_primary, grounded_control = detector.raw_scores(grounded)
    switched_primary, switched_control = detector.raw_scores(switched)

    assert switched_primary[token] > grounded_primary[token] + 5.0
    np.testing.assert_allclose(
        switched_control[token], grounded_control[token], atol=1e-6
    )


def test_common_grounded_focus_is_not_anomaly_merely_because_it_is_narrow():
    detector = TransitionDetector(prototype_count=2).fit(references())
    broad = detector.raw_score(record("test-broad", np.zeros(12)))
    grounded = detector.raw_score(record("test-focus", np.full(12, 4.0)))

    assert np.nanmax(broad) < 0.1
    assert np.nanmax(grounded) < 2.0


def test_sparse_conditioning_cells_use_the_nearest_populated_reference():
    detector = TransitionDetector(prototype_count=2).fit(references())
    detector.calibrate(
        (
            record("calibration-broad", np.zeros(30), prompt=200),
            record("calibration-focus", np.full(30, 4.0), prompt=200),
        )
    )
    query = record("query", np.zeros(30), prompt=10)

    assert np.isfinite(detector.raw_score(query)[HISTORY:]).all()
    assert np.isfinite(detector.score(query)[HISTORY:]).all()


def test_full_structural_change_reaches_the_primary_score():
    detector = TransitionDetector(prototype_count=2).fit(references())
    token = 7
    baseline = record("baseline", np.zeros(12))
    baseline_score = detector.raw_score(baseline)[token]

    for name in BLOCK_NAMES:
        tensor = getattr(baseline.sequence, name).clone()
        tensor[token] = 2.0
        changed = GraphRecord(
            f"changed-{name}",
            baseline.prompt_length,
            replace(baseline.sequence, **{name: tensor}),
        )
        assert detector.raw_score(changed)[token] > baseline_score


def test_invalid_rows_and_two_context_rows_are_excluded():
    detector = TransitionDetector(prototype_count=2).fit(references())
    valid = np.ones(7, dtype=bool)
    valid[4] = False
    item = GraphRecord("invalid", 100, graph(np.zeros(7), valid=valid))
    score = detector.raw_score(item)

    assert np.isnan(score[:HISTORY]).all()
    assert np.isnan(score[4])
    assert np.isfinite(score[[2, 3, 5, 6]]).all()


def test_source_ecdf_gives_each_source_equal_total_weight():
    many = [record("many", np.zeros(3)) for _ in range(10)]
    one = record("one", np.zeros(3))
    records = (*many, one)
    scores = [np.array([np.nan, np.nan, 0.0]) for _ in many]
    scores.append(np.array([np.nan, np.nan, 10.0]))
    edges = prompt_length_edges(records)
    calibrator = SourceECDF.fit(records, scores, edges)

    result = calibrator.transform(
        record("held-out", np.zeros(3)), np.array([np.nan, np.nan, 0.0])
    )
    np.testing.assert_allclose(result[2], 0.5)


def test_calibration_sources_are_disjoint_from_prototype_sources():
    detector = TransitionDetector(prototype_count=2).fit(references())

    with np.testing.assert_raises_regex(ValueError, "must be disjoint"):
        detector.calibrate((references()[0],))


def test_fit_is_deterministic_and_keeps_actual_windows():
    forward = TransitionDetector(prototype_count=2).fit(references())
    backward = TransitionDetector(prototype_count=2).fit(tuple(reversed(references())))

    assert forward.reference.keys() == backward.reference.keys()
    for key in forward.reference:
        left = forward.reference[key]
        right = backward.reference[key]
        np.testing.assert_allclose(left.weight, right.weight)
        for name in BLOCK_NAMES:
            np.testing.assert_array_equal(left.tensor[name], right.tensor[name])


def test_save_load_preserves_raw_and_calibrated_scores(tmp_path):
    detector = TransitionDetector(prototype_count=2).fit(references())
    calibration = (
        record("calibration-0", np.zeros(12)),
        record("calibration-1", np.full(12, 4.0)),
    )
    detector.calibrate(calibration)
    query = record("query", np.r_[np.zeros(7), 4.0, np.zeros(4)])
    output = tmp_path / "detector.npz"
    detector.save(output)
    restored = TransitionDetector.load(output)

    np.testing.assert_allclose(
        restored.raw_score(query), detector.raw_score(query), equal_nan=True
    )
    np.testing.assert_allclose(
        restored.score(query), detector.score(query), equal_nan=True
    )
    np.testing.assert_allclose(
        restored.independent_score(query),
        detector.independent_score(query),
        equal_nan=True,
    )
