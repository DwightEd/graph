"""Configuration for the label-free graph structure and recoverability audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraphAuditConfig:
    """Structural statistics, masking, and output settings."""

    prompt_bins: int = 16
    coalition_top_sources: int = 12
    source_mask_fraction: float = 0.25
    channel_mask_fraction: float = 0.25
    minimum_sources_for_recovery: int = 4
    minimum_channels_for_recovery: int = 4
    block_rows: int = 8192
    random_seed: int = 20260822
    show_progress: bool = True

    def validate(self) -> None:
        positive = (
            self.prompt_bins,
            self.coalition_top_sources,
            self.minimum_sources_for_recovery,
            self.minimum_channels_for_recovery,
            self.block_rows,
        )
        if min(positive) < 1:
            raise ValueError("integer configuration values must be positive")
        if not 0.0 < self.source_mask_fraction < 1.0:
            raise ValueError("source_mask_fraction must be in (0, 1)")
        if not 0.0 < self.channel_mask_fraction < 1.0:
            raise ValueError("channel_mask_fraction must be in (0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
