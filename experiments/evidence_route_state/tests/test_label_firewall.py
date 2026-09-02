from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch

from experiments.evidence_route_state.detector import GraphRecord, TransitionDetector
from experiments.evidence_route_state.graph import GraphSequence


def graph(trajectory: np.ndarray) -> GraphSequence:
    values = np.asarray(trajectory, dtype=np.float32)
    tokens = len(values)

    def block(shape: tuple[int, ...]) -> torch.Tensor:
        result = np.zeros((tokens, *shape), dtype=np.float32)
        result.reshape(tokens, -1)[:] = values[:, None]
        return torch.from_numpy(result)

    position = torch.arange(30, 30 + tokens)
    return GraphSequence(
        query_position=position - 1,
        prediction_position=position,
        node_embedding=block((4, 3)),
        residual_gram=block((3, 4, 4)),
        head_write_gram=block((2, 2, 4, 4)),
        route_topology=block((2, 2, 4, 7)),
        mlp_relation=block((2, 5)),
        margin_contribution=block((4,)),
        valid=torch.ones(tokens, dtype=torch.bool),
    )


class PoisonRecord:
    """Duck-typed graph record whose label cannot be read by the detector."""

    def __init__(self, source_id: str, trajectory: np.ndarray, hallucinated: bool):
        self.source_id = source_id
        self.prompt_length = 100
        self.sequence = graph(trajectory)
        self._hallucinated = hallucinated

    @property
    def hallucination_labels(self):
        raise AssertionError("detector opened labels before post-hoc evaluation")


def records(prefix: str) -> tuple[PoisonRecord, ...]:
    return (
        PoisonRecord(f"{prefix}-broad", np.zeros(12), False),
        PoisonRecord(f"{prefix}-focus", np.full(12, 4.0), True),
    )


def test_graph_and_detector_contracts_have_no_label_field():
    assert "label" not in {field.name for field in fields(GraphSequence)}
    assert "label" not in {field.name for field in fields(GraphRecord)}


def test_flipping_poison_labels_cannot_change_graph_score():
    detector = TransitionDetector(prototype_count=2).fit(records("reference"))
    detector.calibrate(records("calibration"))
    trajectory = np.r_[np.zeros(7), 4.0, np.zeros(4)]
    correct = PoisonRecord("query", trajectory, False)
    hallucinated = PoisonRecord("query", trajectory, True)

    np.testing.assert_allclose(
        detector.score(correct),
        detector.score(hallucinated),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        detector.independent_score(correct),
        detector.independent_score(hallucinated),
        equal_nan=True,
    )


def test_reference_and_calibration_sources_must_be_disjoint():
    reference = records("reference")
    detector = TransitionDetector(prototype_count=2).fit(reference)

    with np.testing.assert_raises_regex(ValueError, "must be disjoint"):
        detector.calibrate((reference[0],))

    calibration = records("calibration")
    detector.calibrate(calibration)
    assert detector.reference_sources.isdisjoint(
        record.source_id for record in calibration
    )
