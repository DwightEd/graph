"""Numerical settings for the attention phenomenology audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PhenomenologyConfig:
    """Shared settings for routing, geometry, null models, and references."""

    prompt_bins: int = 4
    rr_lag_bins: int = 8
    lid_neighbors: int = 8
    transition_projections: int = 12
    anchor_count: int = 8
    recent_lag_max: int = 4
    causal_position_bins: int = 10
    block_rows: int = 8192
    random_seed: int = 20260819
    epsilon: float = 1e-8

    @property
    def known_role_count(self) -> int:
        return self.prompt_bins + self.rr_lag_bins + 1

    @property
    def role_count(self) -> int:
        return self.known_role_count + 1

    @property
    def self_role(self) -> int:
        return self.prompt_bins + self.rr_lag_bins

    @property
    def unresolved_role(self) -> int:
        return self.self_role + 1

    @property
    def role_names(self) -> tuple[str, ...]:
        prompt = tuple(f"prompt_bin_{index}" for index in range(self.prompt_bins))
        history = tuple(f"rr_lag_bin_{index}" for index in range(self.rr_lag_bins))
        return prompt + history + ("self", "unresolved")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
