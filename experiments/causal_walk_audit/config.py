"""Configuration for the causal-walk validation suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WalkAuditConfig:
    block_rows: int = 8192
    max_anchors: int = 12
    prompt_chunk_tokens: int = 32
    train_reservoir_rows: int = 100_000
    ridge_alpha: float = 1.0
    score_horizon: int = 4
    minimum_anchor_mass: float = 1e-3
    anchor_shuffle_replicates: int = 8
    bootstrap_replicates: int = 500
    permutation_replicates: int = 199
    random_seed: int = 20260824
    show_progress: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
