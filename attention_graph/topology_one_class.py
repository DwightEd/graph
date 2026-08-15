"""Hierarchical label-free scoring of causal-attention topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .causal_topology import TopologyEncoding
from .one_class import CalibratedMaxFusion, OneClassConfig, OneClassReference


@dataclass(frozen=True)
class TopologyScoreResult:
    """The fixed, interpretable hierarchy of topology anomaly scores."""

    scores: dict[str, np.ndarray]


def atomic_blocks(encoding: TopologyEncoding) -> dict[str, np.ndarray]:
    """Return one [tokens, layers*heads] block for every topology coordinate."""
    groups = {
        "balance_scale": encoding.balance_log_scale,
        "attention_marginals": encoding.attention_marginals,
        "retained_support": encoding.retained_support,
        "prompt_provenance": encoding.prompt_provenance,
        "rr_one_hop_exact": encoding.rr_one_hop,
        "rr_two_hop_exact": encoding.rr_two_hop,
        "rr_one_hop_lag_rewired": encoding.rewired_rr_one_hop,
        "rr_two_hop_lag_rewired": encoding.rewired_rr_two_hop,
    }
    blocks: dict[str, np.ndarray] = {}
    for group, values in groups.items():
        for coordinate in range(values.shape[-1]):
            blocks[f"{group}:{coordinate}"] = _channels(values[..., coordinate])
    return blocks


def _channels(values: torch.Tensor) -> np.ndarray:
    return values.detach().reshape(values.shape[0], -1).float().cpu().numpy()


def _position_bins(position: np.ndarray, count: int) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    return np.minimum((position * count).astype(np.int16), count - 1)


def _bin_positions(bins: np.ndarray, count: int) -> np.ndarray:
    return (np.asarray(bins, dtype=np.float32) + .5) / count


class TopologyOneClassModel:
    """Fit independent atomic references, then fuse only calibrated scores."""

    _final_score_names = (
        "attention_marginals",
        "retained_support",
        "balance_scale",
        "prompt_topology",
        "rr_one_hop_exact",
        "rr_two_hop_exact",
        "rr_multihop_exact",
        "rr_multihop_lag_rewired",
        "causal_topology_exact",
        "causal_topology_lag_rewired",
        "full_signal",
    )

    def __init__(self, config: OneClassConfig):
        self.config = config

    def fit(
        self,
        fit_encoding: TopologyEncoding,
        fit_position: np.ndarray,
        calibration_encoding: TopologyEncoding,
        calibration_position: np.ndarray,
    ) -> "TopologyOneClassModel":
        return self.fit_blocks(
            atomic_blocks(fit_encoding),
            _position_bins(fit_position, self.config.position_bins),
            atomic_blocks(calibration_encoding),
            _position_bins(calibration_position, self.config.position_bins),
        )

    def fit_blocks(
        self,
        fit_blocks: dict[str, np.ndarray],
        fit_bins: np.ndarray,
        calibration_blocks: dict[str, np.ndarray],
        calibration_bins: np.ndarray,
    ) -> "TopologyOneClassModel":
        """Fit atomic references from aligned train-only fit/calibration rows."""
        return self.fit_loaders(
            tuple(fit_blocks), fit_bins, calibration_bins,
            fit_blocks.__getitem__, calibration_blocks.__getitem__,
        )

    def fit_loaders(
        self,
        block_names: tuple[str, ...],
        fit_bins: np.ndarray,
        calibration_bins: np.ndarray,
        fit_block: Callable[[str], np.ndarray],
        calibration_block: Callable[[str], np.ndarray],
    ) -> "TopologyOneClassModel":
        """Fit one atomic block at a time to bound peak host memory."""
        self.references = {}
        calibration_scores = {}
        for name in block_names:
            reference = OneClassReference(self.config).fit(
                fit_block(name), fit_bins,
                calibration_block(name), calibration_bins,
            )
            self.references[name] = reference
            calibration_scores[name] = reference.calibration_score
        self.fusions = {}
        calibration_values = dict(calibration_scores)
        for name, children in self._hierarchy(calibration_scores).items():
            fusion = CalibratedMaxFusion().fit(
                {child: calibration_values[child] for child in children}
            )
            self.fusions[name] = fusion
            calibration_values[name] = fusion.transform(
                {child: calibration_values[child] for child in children}
            )
        self._calibration_scores = {
            name: calibration_values[name].copy() for name in self._final_score_names
        }
        return self

    def transform(
        self, encoding: TopologyEncoding, position: np.ndarray
    ) -> TopologyScoreResult:
        return self.transform_blocks(atomic_blocks(encoding), position)

    def transform_blocks(
        self, blocks: dict[str, np.ndarray], position: np.ndarray
    ) -> TopologyScoreResult:
        """Score aligned blocks without changing their fitted references."""
        atomic_scores = {
            name: reference.transform(blocks[name], position).score
            for name, reference in self.references.items()
        }
        values = dict(atomic_scores)
        for name, children in self._hierarchy(atomic_scores).items():
            values[name] = self.fusions[name].transform(
                {child: values[child] for child in children}
            )
        return TopologyScoreResult(
            {name: values[name] for name in self._final_score_names}
        )

    def calibration_scores(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._calibration_scores.items()}

    def threshold(self, score_name: str, quantile: float = .95) -> float:
        return float(np.quantile(self._calibration_scores[score_name], quantile))

    def state(self) -> dict[str, np.ndarray]:
        """Flatten references into an NPZ-ready mapping."""
        state: dict[str, np.ndarray] = {}
        for name, reference in self.references.items():
            state.update({
                f"atomic/{name}/{key}": value for key, value in reference.state().items()
            })
        for name, fusion in self.fusions.items():
            state.update({
                f"fusion/{name}/{key}": value for key, value in fusion.state().items()
            })
        state.update({
            f"calibration/{name}": values
            for name, values in self._calibration_scores.items()
        })
        return state

    @staticmethod
    def _hierarchy(atomic_scores: dict[str, np.ndarray]) -> dict[str, tuple[str, ...]]:
        def coordinates(prefix: str) -> tuple[str, ...]:
            return tuple(name for name in atomic_scores if name.startswith(f"{prefix}:"))

        return {
            "balance_scale": coordinates("balance_scale"),
            "attention_marginals": coordinates("attention_marginals"),
            "retained_support": coordinates("retained_support"),
            "prompt_topology": coordinates("prompt_provenance"),
            "rr_one_hop_exact": coordinates("rr_one_hop_exact"),
            "rr_two_hop_exact": coordinates("rr_two_hop_exact"),
            "rr_multihop_exact": ("rr_one_hop_exact", "rr_two_hop_exact"),
            "rr_one_hop_lag_rewired": coordinates("rr_one_hop_lag_rewired"),
            "rr_two_hop_lag_rewired": coordinates("rr_two_hop_lag_rewired"),
            "rr_multihop_lag_rewired": (
                "rr_one_hop_lag_rewired", "rr_two_hop_lag_rewired"
            ),
            "causal_topology_exact": (
                "prompt_topology", "rr_multihop_exact",
            ),
            "causal_topology_lag_rewired": (
                "prompt_topology", "rr_multihop_lag_rewired",
            ),
            "full_signal": (
                "attention_marginals", "balance_scale", "retained_support",
                "causal_topology_exact",
            ),
        }
