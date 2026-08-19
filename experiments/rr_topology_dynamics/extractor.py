"""Orchestrate the focused routing-attractor feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.spectral_feasibility.representations import response_position_bin

from .attractor import AttractorFeatureExtractor
from .routing_state import RoutingStateExtractor


@dataclass(frozen=True)
class TopologyDynamicsConfig:
    block_rows: int = 8192
    position_bins: int = 8
    recent_lag_max: int = 4
    spectral_top_k: int = 5
    epsilon: float = 1e-8

    def validate(self) -> None:
        if min(
            int(self.block_rows),
            int(self.position_bins),
            int(self.recent_lag_max),
            int(self.spectral_top_k),
        ) < 1:
            raise ValueError("topology-dynamics integer settings must be positive")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class TopologyExtraction:
    feature_names: tuple[str, ...]
    features: np.ndarray
    control_names: tuple[str, ...]
    controls: np.ndarray
    position_bin: np.ndarray


class TopologyDynamicsExtractor:
    """Extract the predeclared attractor signals from one research sample."""

    def __init__(
        self,
        config: TopologyDynamicsConfig | None = None,
    ) -> None:
        self.config = TopologyDynamicsConfig() if config is None else config
        self.config.validate()
        self.routing = RoutingStateExtractor(block_rows=self.config.block_rows)
        self.attractor = AttractorFeatureExtractor(
            recent_lag_max=self.config.recent_lag_max,
            epsilon=self.config.epsilon,
        )

    def extract(self, sample) -> TopologyExtraction:
        state = self.routing.extract(sample)
        features = self.attractor.extract(state)
        position_bin = np.asarray(
            [
                response_position_bin(
                    token, state.response_count, self.config.position_bins
                )
                for token in range(state.response_count)
            ],
            dtype=np.int16,
        )
        return TopologyExtraction(
            feature_names=features.names,
            features=features.values.detach().cpu().numpy().astype(np.float32),
            control_names=features.control_names,
            controls=features.controls.detach().cpu().numpy().astype(np.float32),
            position_bin=position_bin,
        )
