"""Dataset-independent input. None labels means unreviewed, [] reviewed negative."""

from dataclasses import dataclass, field


@dataclass
class Example:
    id: str
    source_id: str
    prompt: str
    evidence: list[dict] = field(default_factory=list)  # {id, start, end} in prompt
    response: str | None = None
    labels: list[dict] | None = None  # None = unreviewed; [] = reviewed negative
    metadata: dict = field(default_factory=dict)
