"""Validated records at the factorial-effect input seam."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


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
        if not all(isinstance(value, (int, float)) and isfinite(value) for value in values):
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

