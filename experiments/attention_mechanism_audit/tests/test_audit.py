from dataclasses import dataclass

import numpy as np
import pytest

from experiments.attention_mechanism_audit.audit import (
    AuditArtifact,
    RawMargins,
    UnidentifiablePrior,
    audit_pair,
    load_artifact,
    save_artifact,
)


@dataclass(frozen=True)
class Span:
    start: int
    stop: int


@dataclass(frozen=True)
class Branch:
    input_ids: tuple[int, ...]
    predictor_index: int


@dataclass(frozen=True)
class Pair:
    sample_id: str
    source_id: str
    question_only: Branch
    context_a: Branch
    context_b: Branch
    relevant_span: Span
    irrelevant_span: Span
    history_span: Span
    candidate_a_token_id: int
    candidate_b_token_id: int


class RecordingReplay:
    def __init__(self, question_raw):
        self.calls = []
        self.question_raw = question_raw
        self.orientation = -1.0 if question_raw > 0 else 1.0
        self.checkpoint = "frozen-test-model"

    def score_margin(self, input_ids, predictor_index, candidate_b, candidate_a):
        self.calls.append(
            ("full", tuple(input_ids), predictor_index, candidate_b, candidate_a)
        )
        return {1: self.question_raw, 2: -3.0, 3: 3.0}[input_ids[0]]

    def score_without_prompt_sources_margin(
        self, input_ids, predictor_index, candidate_b, candidate_a, source_positions
    ):
        source_positions = tuple(source_positions)
        self.calls.append(("prompt", source_positions))
        desired = {(1, 2): -1.5, (3, 4): -0.2}[source_positions]
        return desired / self.orientation

    def score_without_history_margin(
        self,
        input_ids,
        predictor_index,
        candidate_b,
        candidate_a,
        history_start,
        history_stop,
    ):
        self.calls.append(("history", history_start, history_stop))
        return 3.75 / self.orientation

    def capture_history_kv(
        self,
        input_ids,
        predictor_index,
        candidate_b,
        candidate_a,
        history_start,
        history_stop,
    ):
        self.calls.append(
            (
                "capture",
                tuple(input_ids),
                predictor_index,
                candidate_b,
                candidate_a,
                history_start,
                history_stop,
            )
        )
        return "prior-history-kv", {2: -3.0, 3: 3.0}[input_ids[0]]

    def score_hybrid_history_margin(
        self, input_ids, predictor_index, candidate_b, candidate_a, history_kv
    ):
        self.calls.append(("hybrid", history_kv))
        return 2.25 / self.orientation


@pytest.mark.parametrize(
    ("question_raw", "prior_marker"),
    [(2.0, 3), (-2.0, 2)],
)
def test_audit_pair_dynamically_orients_prior_and_runs_fixed_interventions(
    question_raw, prior_marker
):
    pair = Pair(
        sample_id="sample",
        source_id="source",
        question_only=Branch((1, 9), 1),
        context_a=Branch((2, 8, 7, 6, 5, 4, 3, 2, 1), 8),
        context_b=Branch((3, 8, 7, 6, 5, 4, 3, 2, 1), 8),
        relevant_span=Span(1, 3),
        irrelevant_span=Span(3, 5),
        history_span=Span(5, 9),
        candidate_a_token_id=10,
        candidate_b_token_id=11,
    )
    replay = RecordingReplay(question_raw)

    row = audit_pair(pair, replay)

    assert row.sample_id == "sample"
    assert row.prior_is_b is (question_raw > 0)
    assert row.margins == RawMargins(-2.0, -3.0, 3.0, -1.5, -0.2, 3.75, 2.25)
    assert row.margins.relevant_gain == 4.5
    assert row.margins.select_contrast == 1.3
    assert row.margins.history_prior_support == 0.75
    assert row.margins.history_evidence_relay == 0.75
    assert row.margins.question_prior_strength == 2.0
    assert row.margins.prior_capture == -3.0
    assert ("prompt", (1, 2)) in replay.calls
    assert ("prompt", (3, 4)) in replay.calls
    assert ("history", 5, 9) in replay.calls
    assert any(
        call[0] == "capture" and call[1][0] == prior_marker
        for call in replay.calls
    )
    assert ("hybrid", "prior-history-kv") in replay.calls
    assert len(replay.calls) == 7


def test_exact_question_only_tie_is_not_assigned_an_arbitrary_prior():
    pair = Pair(
        sample_id="tie",
        source_id="source",
        question_only=Branch((1, 9), 1),
        context_a=Branch((2, 8, 7), 2),
        context_b=Branch((3, 8, 7), 2),
        relevant_span=Span(0, 1),
        irrelevant_span=Span(1, 2),
        history_span=Span(2, 3),
        candidate_a_token_id=10,
        candidate_b_token_id=11,
    )
    replay = RecordingReplay(0.0)

    with pytest.raises(UnidentifiablePrior, match="question-only prior is tied"):
        audit_pair(pair, replay)

    assert replay.calls == [("full", (1, 9), 1, 11, 10)]


def test_fixed_artifact_round_trip_contains_raw_margins_and_derived_mechanisms(
    tmp_path,
):
    rows = [
        (
            "a",
            "source-1",
            10,
            11,
            True,
            RawMargins(-2.0, -3.0, -0.5, -1.5, -0.2, 0.25, -1.25),
        ),
        (
            "b",
            "source-2",
            20,
            21,
            False,
            RawMargins(-1.0, -2.0, 2.0, 0.5, 1.5, 1.0, 0.0),
        ),
    ]
    from experiments.attention_mechanism_audit.audit import AuditRow

    artifact = AuditArtifact.from_rows(AuditRow(*row) for row in rows)
    path = tmp_path / "audit.npz"
    manifest = save_artifact(artifact, path)
    loaded = load_artifact(path)

    assert manifest["labels_used"] is False
    np.testing.assert_array_equal(loaded.sample_id, ["a", "b"])
    np.testing.assert_allclose(loaded.relevant_gain, [1.0, 1.5])
    np.testing.assert_allclose(loaded.select_contrast, [1.3, 1.0])
    np.testing.assert_allclose(loaded.history_prior_support, [0.75, -1.0])
    np.testing.assert_allclose(loaded.history_evidence_relay, [0.75, 2.0])
    np.testing.assert_allclose(loaded.question_prior_strength, [2.0, 1.0])
    np.testing.assert_allclose(loaded.prior_capture, [0.5, -2.0])

    with np.load(path, allow_pickle=False) as saved:
        assert "feature_names" not in saved.files
        assert "label" not in saved.files
        assert "prior_capture" in saved.files
        assert "counterevidence_override" not in saved.files
