"""Strict token-level inputs for the grounding-control audit.

The audit consumes prepared causal pairs.  It never rebuilds prompts or infers
semantic roles from text.  ``context_a`` and ``context_b`` are symmetric,
position-aligned factual controls; ``question_only`` may be shorter because it
contains no evidence context, but it must end in the same neutral history.
Neither candidate is named as the parametric prior in the input artifact.

``relevant_span`` and ``irrelevant_span`` are matched answer-bearing value
slots, not whole fact sentences.  By construction the relevant value in
``context_a`` supports candidate A and the relevant value in ``context_b``
supports candidate B; the irrelevant slot carries the swapped control value.
That semantic contract is curated upstream and is never guessed from token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _record(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


@dataclass(frozen=True)
class TokenSpan:
    """A non-empty half-open token interval ``[start, stop)``."""

    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def validate(self, token_count: int, name: str) -> "TokenSpan":
        if not 0 <= self.start < self.stop <= token_count:
            raise ValueError(f"{name} must be a non-empty span inside the context")
        return self

    @classmethod
    def from_json(cls, value: object, name: str) -> "TokenSpan":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{name} must be [start, stop]")
        return cls(
            _integer(value[0], f"{name}[0]"),
            _integer(value[1], f"{name}[1]"),
        )


@dataclass(frozen=True)
class AuditBranch:
    """One directly supplied model prefix and its candidate-logit row."""

    input_ids: tuple[int, ...]
    predictor_index: int

    def validate(self, name: str) -> "AuditBranch":
        if not self.input_ids or any(
            type(token) is not int or token < 0 for token in self.input_ids
        ):
            raise ValueError(f"{name}.input_ids must be non-negative token IDs")
        if self.predictor_index != len(self.input_ids) - 1:
            raise ValueError(
                f"{name}.predictor_index must be the final supplied token"
            )
        return self

    @classmethod
    def from_json(cls, value: object, name: str) -> "AuditBranch":
        item = _record(value, {"input_ids", "predictor_index"}, name)
        input_ids = item["input_ids"]
        if not isinstance(input_ids, list):
            raise ValueError(f"{name}.input_ids must be a token ID list")
        return cls(
            input_ids=tuple(input_ids),
            predictor_index=_integer(
                item["predictor_index"], f"{name}.predictor_index"
            ),
        ).validate(name)


@dataclass(frozen=True)
class AuditPair:
    """One matched answer-value swap used by all three mechanism tests."""

    sample_id: str
    source_id: str
    question_only: AuditBranch
    context_a: AuditBranch
    context_b: AuditBranch
    relevant_span: TokenSpan
    irrelevant_span: TokenSpan
    history_span: TokenSpan
    candidate_a_token_id: int
    candidate_b_token_id: int
    decision_prefix_is_neutral: bool

    def validate(self) -> "AuditPair":
        if not self.sample_id or not self.source_id:
            raise ValueError("sample_id and source_id must be non-empty strings")

        branches = {
            "question_only": self.question_only,
            "context_a": self.context_a,
            "context_b": self.context_b,
        }
        for name, branch in branches.items():
            branch.validate(name)

        context_a = self.context_a.input_ids
        context_b = self.context_b.input_ids
        if len(context_a) != len(context_b):
            raise ValueError("context_a and context_b must be equal length")
        if (
            self.context_a.predictor_index
            != self.context_b.predictor_index
        ):
            raise ValueError("paired contexts must use the same predictor position")

        token_count = len(context_a)
        spans = {
            "relevant_span": self.relevant_span.validate(
                token_count, "relevant_span"
            ),
            "irrelevant_span": self.irrelevant_span.validate(
                token_count, "irrelevant_span"
            ),
            "history_span": self.history_span.validate(token_count, "history_span"),
        }
        occupied: set[int] = set()
        for name, span in spans.items():
            positions = set(range(span.start, span.stop))
            if occupied & positions:
                raise ValueError(f"{name} overlaps another audit span")
            occupied |= positions

        evidence_stop = max(self.relevant_span.stop, self.irrelevant_span.stop)
        if evidence_stop > self.history_span.start:
            raise ValueError("evidence spans must precede the answer history")
        if len(self.relevant_span) != len(self.irrelevant_span):
            raise ValueError("relevant and irrelevant spans must have equal capacity")
        if self.history_span.stop != self.context_a.predictor_index + 1:
            raise ValueError("history_span must end at the predictor position")
        if self.history_span.start == self.context_a.predictor_index:
            raise ValueError("history_span must include a key before the predictor")
        if self.decision_prefix_is_neutral is not True:
            raise ValueError(
                "the shared decision/history prefix must be declared neutral"
            )

        history = context_a[self.history_span.start : self.history_span.stop]
        if context_b[self.history_span.start : self.history_span.stop] != history:
            raise ValueError("paired contexts must contain the same answer history")
        question_history_start = self.question_only.predictor_index + 1 - len(history)
        if (
            question_history_start < 0
            or self.question_only.input_ids[
                question_history_start : self.question_only.predictor_index + 1
            ]
            != history
        ):
            raise ValueError("question_only must end in the same answer history")

        editable = set(range(self.relevant_span.start, self.relevant_span.stop))
        editable.update(range(self.irrelevant_span.start, self.irrelevant_span.stop))
        changed_outside_values = any(
            context_a[index] != context_b[index]
            for index in range(token_count)
            if index not in editable
        )
        if changed_outside_values:
            raise ValueError("paired contexts may differ only inside the value slots")
        relevant_a = context_a[
            self.relevant_span.start : self.relevant_span.stop
        ]
        relevant_b = context_b[
            self.relevant_span.start : self.relevant_span.stop
        ]
        irrelevant_a = context_a[
            self.irrelevant_span.start : self.irrelevant_span.stop
        ]
        irrelevant_b = context_b[
            self.irrelevant_span.start : self.irrelevant_span.stop
        ]
        if relevant_a != irrelevant_b or irrelevant_a != relevant_b:
            raise ValueError(
                "context_a and context_b must exactly exchange the two value slots"
            )

        candidates = (
            self.candidate_a_token_id,
            self.candidate_b_token_id,
        )
        if any(type(token) is not int or token < 0 for token in candidates):
            raise ValueError("candidate token IDs must be non-negative integers")
        if candidates[0] == candidates[1]:
            raise ValueError("first-divergence candidate token IDs must differ")
        return self

    @classmethod
    def from_json(cls, value: object) -> "AuditPair":
        keys = {
            "sample_id",
            "source_id",
            "question_only",
            "context_a",
            "context_b",
            "relevant_span",
            "irrelevant_span",
            "history_span",
            "candidate_a_token_id",
            "candidate_b_token_id",
            "decision_prefix_is_neutral",
        }
        item = _record(value, keys, "audit pair")
        if not isinstance(item["sample_id"], str) or not isinstance(
            item["source_id"], str
        ):
            raise ValueError("sample_id and source_id must be strings")
        return cls(
            sample_id=item["sample_id"],
            source_id=item["source_id"],
            question_only=AuditBranch.from_json(
                item["question_only"], "question_only"
            ),
            context_a=AuditBranch.from_json(
                item["context_a"], "context_a"
            ),
            context_b=AuditBranch.from_json(
                item["context_b"], "context_b"
            ),
            relevant_span=TokenSpan.from_json(item["relevant_span"], "relevant_span"),
            irrelevant_span=TokenSpan.from_json(
                item["irrelevant_span"], "irrelevant_span"
            ),
            history_span=TokenSpan.from_json(item["history_span"], "history_span"),
            candidate_a_token_id=_integer(
                item["candidate_a_token_id"], "candidate_a_token_id"
            ),
            candidate_b_token_id=_integer(
                item["candidate_b_token_id"], "candidate_b_token_id"
            ),
            decision_prefix_is_neutral=item["decision_prefix_is_neutral"],
        ).validate()


def load_pairs(path: str | Path) -> list[AuditPair]:
    """Load the exact ``audit_pairs.jsonl`` contract without text fallbacks."""

    pairs: list[AuditPair] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                pair = AuditPair.from_json(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid audit pair on line {line_number}: {error}"
                ) from error
            if pair.sample_id in seen:
                raise ValueError(
                    f"duplicate sample_id on line {line_number}: {pair.sample_id}"
                )
            seen.add(pair.sample_id)
            pairs.append(pair)
    return pairs
