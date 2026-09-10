"""Validated records at the factorial-effect input seam."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

EVENT_SCHEMA = "control-graph/factorial-event@1"
EVENT_FIELDS = {"schema", "event_id", "source_id", "split", "relation", "margins"}


@dataclass(frozen=True)
class FactorialMargins:
    """A-minus-B margins from the two-source by two-prefix experiment."""

    onset_a: float
    onset_b: float
    world_a_after_a: float
    world_a_after_b: float
    world_b_after_a: float
    world_b_after_b: float

    def __post_init__(self) -> None:
        values = tuple(vars(self).values())
        if not all(type(value) in {int, float} and isfinite(value) for value in values):
            raise ValueError("factorial margins must be finite numbers")


@dataclass(frozen=True)
class FactorialEvent:
    """One relation event with provenance and its measured margins."""

    event_id: str
    source_id: str
    split: str
    relation: str
    margins: FactorialMargins

    def __post_init__(self) -> None:
        text = (self.event_id, self.source_id, self.split, self.relation)
        if any(not isinstance(value, str) or not value.strip() for value in text):
            raise ValueError("event identity fields must be non-empty strings")


def load_factorial_events(path: str | Path) -> tuple[FactorialEvent, ...]:
    """Load label-free factorial records and reject undeclared fields."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"factorial event JSONL does not exist: {path}")
    events = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        unexpected = set(record).difference(EVENT_FIELDS)
        missing = EVENT_FIELDS.difference(record)
        if unexpected or missing:
            raise ValueError(
                f"{path}:{line_number} has unexpected fields {sorted(unexpected)} "
                f"or missing fields {sorted(missing)}"
            )
        if record["schema"] != EVENT_SCHEMA:
            raise ValueError(f"{path}:{line_number} has an unsupported schema")
        margins = record["margins"]
        if not isinstance(margins, dict) or set(margins) != set(FactorialMargins.__annotations__):
            raise ValueError(f"{path}:{line_number} has invalid factorial margins")
        event = FactorialEvent(
            event_id=record["event_id"],
            source_id=record["source_id"],
            split=record["split"],
            relation=record["relation"],
            margins=FactorialMargins(**margins),
        )
        if event.event_id in seen:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        seen.add(event.event_id)
        events.append(event)
    if not events:
        raise ValueError(f"factorial event JSONL is empty: {path}")
    return tuple(events)
