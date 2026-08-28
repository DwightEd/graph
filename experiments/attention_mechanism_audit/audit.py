"""Fixed causal audit for the SELECT--RELAY--OVERRIDE chain.

Replay first returns the symmetric raw margin

    raw = logit(candidate_B) - logit(candidate_A).

The question-only raw margin determines which candidate is the model prior.
All persisted branches are then oriented as

    M = logit(counter-supported candidate) - logit(prior candidate).

The audit contains no labels, learned probes, feature registry, or graph
reconstruction.  It only runs the seven interventions fixed by the method.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np


AUDIT_SCHEMA = "grounding-control-chain"
AUDIT_VERSION = 1
MARGIN_DEFINITION = "logit(counter_supported_candidate)-logit(question_only_prior)"


class TokenSpanLike(Protocol):
    start: int
    stop: int


class AuditBranchLike(Protocol):
    input_ids: Sequence[int]
    predictor_index: int


class AuditPairLike(Protocol):
    sample_id: str
    source_id: str
    question_only: AuditBranchLike
    context_a: AuditBranchLike
    context_b: AuditBranchLike
    relevant_span: TokenSpanLike
    irrelevant_span: TokenSpanLike
    history_span: TokenSpanLike
    candidate_a_token_id: int
    candidate_b_token_id: int


class CausalMarginReplayLike(Protocol):
    def score_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
    ) -> float: ...

    def score_without_prompt_sources_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        source_positions: Sequence[int],
    ) -> float: ...

    def score_without_history_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_start: int,
        history_stop: int,
    ) -> float: ...

    def capture_history_kv(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_start: int,
        history_stop: int,
    ) -> tuple[object, float]: ...

    def score_hybrid_history_margin(
        self,
        input_ids: Sequence[int],
        predictor_index: int,
        candidate_b_token_id: int,
        candidate_a_token_id: int,
        history_kv: object,
    ) -> float: ...


@dataclass(frozen=True)
class RawMargins:
    """The seven pre-registered causal branches for one fact pair."""

    question_only: float
    prior_context: float
    counter_context: float
    no_relevant: float
    no_irrelevant: float
    no_history: float
    hybrid_history: float

    @property
    def relevant_gain(self) -> float:
        """Positive means the relevant source increases counter-evidence support."""

        return self.counter_context - self.no_relevant

    @property
    def select_contrast(self) -> float:
        """Positive means the relevant source matters more than its control."""

        return self.no_irrelevant - self.no_relevant

    @property
    def history_prior_support(self) -> float:
        """Positive means history pulls the decision toward the prior answer."""

        return self.no_history - self.counter_context

    @property
    def history_evidence_relay(self) -> float:
        """Positive means counter-evidence is carried in the history K/V state."""

        return self.counter_context - self.hybrid_history

    @property
    def question_prior_strength(self) -> float:
        """Positive means the question-only branch prefers the prior answer."""

        return -self.question_only

    @property
    def prior_capture(self) -> float:
        """Positive means the prior still wins under counter-evidence."""

        return -self.counter_context


@dataclass(frozen=True)
class AuditRow:
    sample_id: str
    source_id: str
    candidate_a_token_id: int
    candidate_b_token_id: int
    prior_is_b: bool
    margins: RawMargins


@dataclass(frozen=True)
class AuditArtifact:
    """A fixed-column table; derived mechanism scores are never learned."""

    sample_id: np.ndarray
    source_id: np.ndarray
    candidate_a_token_id: np.ndarray
    candidate_b_token_id: np.ndarray
    prior_is_b: np.ndarray
    margin_question_only: np.ndarray
    margin_prior_context: np.ndarray
    margin_counter_context: np.ndarray
    margin_no_relevant: np.ndarray
    margin_no_irrelevant: np.ndarray
    margin_no_history: np.ndarray
    margin_hybrid_history: np.ndarray

    @classmethod
    def from_rows(cls, rows: Iterable[AuditRow]) -> "AuditArtifact":
        rows = list(rows)
        if not rows:
            raise ValueError("the causal audit produced no rows")
        return cls(
            sample_id=np.asarray([row.sample_id for row in rows], dtype=np.str_),
            source_id=np.asarray([row.source_id for row in rows], dtype=np.str_),
            candidate_a_token_id=np.asarray(
                [row.candidate_a_token_id for row in rows], dtype=np.int64
            ),
            candidate_b_token_id=np.asarray(
                [row.candidate_b_token_id for row in rows], dtype=np.int64
            ),
            prior_is_b=np.asarray([row.prior_is_b for row in rows], dtype=np.bool_),
            margin_question_only=np.asarray(
                [row.margins.question_only for row in rows], dtype=np.float64
            ),
            margin_prior_context=np.asarray(
                [row.margins.prior_context for row in rows], dtype=np.float64
            ),
            margin_counter_context=np.asarray(
                [row.margins.counter_context for row in rows], dtype=np.float64
            ),
            margin_no_relevant=np.asarray(
                [row.margins.no_relevant for row in rows], dtype=np.float64
            ),
            margin_no_irrelevant=np.asarray(
                [row.margins.no_irrelevant for row in rows], dtype=np.float64
            ),
            margin_no_history=np.asarray(
                [row.margins.no_history for row in rows], dtype=np.float64
            ),
            margin_hybrid_history=np.asarray(
                [row.margins.hybrid_history for row in rows], dtype=np.float64
            ),
        ).validate()

    def validate(self) -> "AuditArtifact":
        columns = (
            self.sample_id,
            self.source_id,
            self.candidate_a_token_id,
            self.candidate_b_token_id,
            self.prior_is_b,
            self.margin_question_only,
            self.margin_prior_context,
            self.margin_counter_context,
            self.margin_no_relevant,
            self.margin_no_irrelevant,
            self.margin_no_history,
            self.margin_hybrid_history,
        )
        if len(self.sample_id) == 0 or any(
            len(column) != len(self.sample_id) for column in columns
        ):
            raise ValueError("causal audit columns are empty or misaligned")
        if len(set(self.sample_id.astype(str).tolist())) != len(self.sample_id):
            raise ValueError("sample_id must be unique")
        if bool((self.candidate_a_token_id == self.candidate_b_token_id).any()):
            raise ValueError("candidate A and B must differ")
        margins = np.column_stack(columns[5:]).astype(np.float64)
        if not np.isfinite(margins).all():
            raise ValueError("all seven causal margins must be finite")
        if bool((self.margin_question_only >= 0.0).any()):
            raise ValueError("question-only margins must be oriented toward the prior")
        return self

    @property
    def relevant_gain(self) -> np.ndarray:
        return self.margin_counter_context - self.margin_no_relevant

    @property
    def select_contrast(self) -> np.ndarray:
        return self.margin_no_irrelevant - self.margin_no_relevant

    @property
    def history_prior_support(self) -> np.ndarray:
        return self.margin_no_history - self.margin_counter_context

    @property
    def history_evidence_relay(self) -> np.ndarray:
        return self.margin_counter_context - self.margin_hybrid_history

    @property
    def question_prior_strength(self) -> np.ndarray:
        return -self.margin_question_only

    @property
    def prior_capture(self) -> np.ndarray:
        return -self.margin_counter_context

    @property
    def prior_candidate_token_id(self) -> np.ndarray:
        return np.where(
            self.prior_is_b,
            self.candidate_b_token_id,
            self.candidate_a_token_id,
        )


class UnidentifiablePrior(ValueError):
    """The question-only branch assigns exactly equal logits to A and B."""


def _positions(span: TokenSpanLike) -> tuple[int, ...]:
    return tuple(range(int(span.start), int(span.stop)))


def audit_pair(pair: AuditPairLike, replay: CausalMarginReplayLike) -> AuditRow:
    """Run exactly the seven branches defined by the causal audit."""

    candidate_a = int(pair.candidate_a_token_id)
    candidate_b = int(pair.candidate_b_token_id)

    def raw_score(branch: AuditBranchLike) -> float:
        return float(
            replay.score_margin(
                branch.input_ids,
                int(branch.predictor_index),
                candidate_b,
                candidate_a,
            )
        )

    raw_question = raw_score(pair.question_only)
    if raw_question == 0.0:
        raise UnidentifiablePrior(
            f"question-only prior is tied for sample {pair.sample_id}"
        )
    prior_is_b = raw_question > 0.0
    prior_context = pair.context_b if prior_is_b else pair.context_a
    counter_context = pair.context_a if prior_is_b else pair.context_b
    orientation = -1.0 if prior_is_b else 1.0

    def oriented(raw_margin: float) -> float:
        return orientation * float(raw_margin)

    prior_history, raw_prior_context = replay.capture_history_kv(
        prior_context.input_ids,
        int(prior_context.predictor_index),
        candidate_b,
        candidate_a,
        int(pair.history_span.start),
        int(pair.history_span.stop),
    )
    margins = RawMargins(
        question_only=oriented(raw_question),
        prior_context=oriented(raw_prior_context),
        counter_context=oriented(raw_score(counter_context)),
        no_relevant=oriented(
            replay.score_without_prompt_sources_margin(
                counter_context.input_ids,
                int(counter_context.predictor_index),
                candidate_b,
                candidate_a,
                _positions(pair.relevant_span),
            )
        ),
        no_irrelevant=oriented(
            replay.score_without_prompt_sources_margin(
                counter_context.input_ids,
                int(counter_context.predictor_index),
                candidate_b,
                candidate_a,
                _positions(pair.irrelevant_span),
            )
        ),
        no_history=oriented(
            replay.score_without_history_margin(
                counter_context.input_ids,
                int(counter_context.predictor_index),
                candidate_b,
                candidate_a,
                int(pair.history_span.start),
                int(pair.history_span.stop),
            )
        ),
        hybrid_history=oriented(
            replay.score_hybrid_history_margin(
                counter_context.input_ids,
                int(counter_context.predictor_index),
                candidate_b,
                candidate_a,
                prior_history,
            )
        ),
    )
    return AuditRow(
        sample_id=str(pair.sample_id),
        source_id=str(pair.source_id),
        candidate_a_token_id=candidate_a,
        candidate_b_token_id=candidate_b,
        prior_is_b=prior_is_b,
        margins=margins,
    )


def _artifact_arrays(artifact: AuditArtifact) -> dict[str, np.ndarray]:
    """Return the complete, non-extensible NPZ schema."""

    return {
        "schema": np.asarray(AUDIT_SCHEMA),
        "version": np.asarray(AUDIT_VERSION, dtype=np.int64),
        "sample_id": artifact.sample_id,
        "source_id": artifact.source_id,
        "candidate_a_token_id": artifact.candidate_a_token_id,
        "candidate_b_token_id": artifact.candidate_b_token_id,
        "prior_is_b": artifact.prior_is_b,
        "margin_question_only": artifact.margin_question_only,
        "margin_prior_context": artifact.margin_prior_context,
        "margin_counter_context": artifact.margin_counter_context,
        "margin_no_relevant": artifact.margin_no_relevant,
        "margin_no_irrelevant": artifact.margin_no_irrelevant,
        "margin_no_history": artifact.margin_no_history,
        "margin_hybrid_history": artifact.margin_hybrid_history,
        "relevant_gain": artifact.relevant_gain,
        "select_contrast": artifact.select_contrast,
        "history_prior_support": artifact.history_prior_support,
        "history_evidence_relay": artifact.history_evidence_relay,
        "question_prior_strength": artifact.question_prior_strength,
        "prior_capture": artifact.prior_capture,
    }


def save_artifact(
    artifact: AuditArtifact,
    path: str | Path,
    *,
    skipped_unidentifiable_sample_ids: Sequence[str] = (),
    model_checkpoint: str | None = None,
) -> dict[str, object]:
    artifact = artifact.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **_artifact_arrays(artifact))
    manifest = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "margin_definition": MARGIN_DEFINITION,
        "samples": len(artifact.sample_id),
        "sources": len(np.unique(artifact.source_id.astype(str))),
        "skipped_unidentifiable_count": len(skipped_unidentifiable_sample_ids),
        "skipped_unidentifiable_sample_ids": list(
            skipped_unidentifiable_sample_ids
        ),
        "labels_used": False,
        "model_checkpoint": model_checkpoint,
        "artifact": str(path),
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_artifact(path: str | Path) -> AuditArtifact:
    with np.load(path, allow_pickle=False) as saved:
        if (
            str(saved["schema"].item()) != AUDIT_SCHEMA
            or int(saved["version"].item()) != AUDIT_VERSION
        ):
            raise ValueError("unsupported causal audit artifact schema")
        artifact = AuditArtifact(
            sample_id=saved["sample_id"].astype(np.str_),
            source_id=saved["source_id"].astype(np.str_),
            candidate_a_token_id=saved["candidate_a_token_id"].astype(np.int64),
            candidate_b_token_id=saved["candidate_b_token_id"].astype(np.int64),
            prior_is_b=saved["prior_is_b"].astype(np.bool_),
            margin_question_only=saved["margin_question_only"].astype(np.float64),
            margin_prior_context=saved["margin_prior_context"].astype(np.float64),
            margin_counter_context=saved["margin_counter_context"].astype(
                np.float64
            ),
            margin_no_relevant=saved["margin_no_relevant"].astype(np.float64),
            margin_no_irrelevant=saved["margin_no_irrelevant"].astype(np.float64),
            margin_no_history=saved["margin_no_history"].astype(np.float64),
            margin_hybrid_history=saved["margin_hybrid_history"].astype(np.float64),
        ).validate()
        expected = _artifact_arrays(artifact)
        for name in (
            "relevant_gain",
            "select_contrast",
            "history_prior_support",
            "history_evidence_relay",
            "question_prior_strength",
            "prior_capture",
        ):
            if not np.array_equal(saved[name], expected[name]):
                raise ValueError(f"persisted derived column is inconsistent: {name}")
    return artifact


def run_audit(
    pairs: Iterable[AuditPairLike],
    replay: CausalMarginReplayLike,
    output: str | Path,
) -> dict[str, object]:
    rows = []
    skipped = []
    for pair in pairs:
        try:
            rows.append(audit_pair(pair, replay))
        except UnidentifiablePrior:
            skipped.append(str(pair.sample_id))
    artifact = AuditArtifact.from_rows(rows)
    return save_artifact(
        artifact,
        output,
        skipped_unidentifiable_sample_ids=skipped,
        model_checkpoint=getattr(replay, "checkpoint", None),
    )
