"""Unsupervised token scores for grounded re-anchor failure.

The detector is deliberately restricted to signals present in schema v8:
head-resolved routing and the all-sample functional context cut.  It does not
pretend that transport budgets are signed, source-specific semantic edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DETECTOR_SCHEMA = 1
RAW_FEATURES = (
    "route_demand",
    "evidence_entry_deficit",
    "context_opposition",
    "context_distribution_js",
    "adoption_deficit",
    "context_target_log_rank",
    "late_evidence_route_loss",
    "predictor_reuse",
    "emitted_token_anchor",
)
SCORE_NAMES = (
    "entry_failure",
    "adoption_failure",
    "override_candidate",
    "online_failure",
    "offline_failure",
)


def _finite_rms(values: np.ndarray, axis) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values * values, 0.0).sum(axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return np.sqrt(result)


def _finite_extreme(arrays: Sequence[np.ndarray], operation: str) -> np.ndarray:
    values = np.stack([np.asarray(value, dtype=np.float64) for value in arrays])
    finite = np.isfinite(values)
    if operation == "min":
        filled = np.where(finite, values, np.inf)
        result = filled.min(axis=0)
    else:
        filled = np.where(finite, values, -np.inf)
        result = filled.max(axis=0)
    result[~finite.any(axis=0)] = np.nan
    return result


def evidence_entry_deficit(
    evidence_transport: np.ndarray,
    route_change: np.ndarray,
    window: int,
) -> np.ndarray:
    """Head-preserving evidence loss at route-changing prediction events."""

    evidence = np.asarray(evidence_transport, dtype=np.float64)
    change = np.asarray(route_change, dtype=np.float64)
    if evidence.ndim != 3 or evidence.shape != change.shape:
        raise ValueError("head routing arrays must share [layer, head, event] shape")
    result = np.full(evidence.shape[-1], np.nan, dtype=np.float64)
    for event in range(window, evidence.shape[-1]):
        history = evidence[..., event - window : event]
        finite_history = np.isfinite(history)
        count = finite_history.sum(axis=-1)
        ordered = np.where(finite_history, history, np.nan)
        reference = np.nanmedian(ordered, axis=-1)
        deficit = np.maximum(reference - evidence[..., event], 0.0)
        weight = change[..., event]
        valid = np.isfinite(deficit) & np.isfinite(weight) & (weight >= 0)
        weight_sum = weight[valid].sum()
        if weight_sum > 0:
            result[event] = np.sqrt(
                np.sum(weight[valid] * deficit[valid] ** 2) / weight_sum
            )
        elif np.any(valid) and np.any(count[valid] > 0):
            result[event] = 0.0
    return result


def late_evidence_route_loss(evidence_transport: np.ndarray) -> np.ndarray:
    """Evidence-route peak lost before the final decoder layer."""

    evidence = np.asarray(evidence_transport, dtype=np.float64)
    if evidence.ndim != 3:
        raise ValueError("evidence transport must have [layer, head, event] shape")
    middle = max(1, evidence.shape[0] // 3)
    late = evidence[middle:]
    finite = np.isfinite(late)
    peak = np.where(finite, late, -np.inf).max(axis=0)
    peak[~finite.any(axis=0)] = np.nan
    loss = np.maximum(peak - evidence[-1], 0.0)
    return _finite_rms(loss, axis=0)


def raw_features(
    result: Mapping[str, np.ndarray], route_window: int
) -> dict[str, np.ndarray]:
    """Extract high-means-failure features from one schema-v8 capture."""

    evidence = np.asarray(result["head_evidence_transport_share"], dtype=np.float64)
    change = np.asarray(result["head_route_change"], dtype=np.float64)
    predictor = np.asarray(result["head_predictor_reuse"], dtype=np.float64)
    emitted = np.asarray(result["head_emitted_token_anchor"], dtype=np.float64)
    count = evidence.shape[-1]
    one_dimensional = {
        "baseline_target_logprob": np.asarray(
            result["baseline_target_logprob"], dtype=np.float64
        ),
        "baseline_entropy": np.asarray(result["baseline_entropy"], dtype=np.float64),
        "context_distribution_js": np.asarray(
            result["context_distribution_js"], dtype=np.float64
        ),
        "context_target_logprob_gain": np.asarray(
            result["context_target_logprob_gain"], dtype=np.float64
        ),
        "context_adoption_margin": np.asarray(
            result["context_adoption_margin"], dtype=np.float64
        ),
        "context_target_log_rank": np.asarray(
            result["context_target_log_rank"], dtype=np.float64
        ),
    }
    if any(value.shape != (count,) for value in one_dimensional.values()):
        raise ValueError("detector fields are not aligned to prediction events")
    if any(value.shape != evidence.shape for value in (change, predictor, emitted)):
        raise ValueError("head detector fields do not share one event geometry")

    return {
        "relative_position": (np.arange(count, dtype=np.float64) + 0.5) / count,
        "baseline_entropy": one_dimensional["baseline_entropy"],
        "baseline_target_logprob": one_dimensional["baseline_target_logprob"],
        "confidence_surprisal": -one_dimensional["baseline_target_logprob"],
        "route_demand": _finite_rms(change, axis=(0, 1)),
        "evidence_entry_deficit": evidence_entry_deficit(
            evidence, change, route_window
        ),
        "context_opposition": -one_dimensional["context_target_logprob_gain"],
        "context_distribution_js": one_dimensional["context_distribution_js"],
        "adoption_deficit": -one_dimensional["context_adoption_margin"],
        "context_target_log_rank": one_dimensional["context_target_log_rank"],
        "late_evidence_route_loss": late_evidence_route_loss(evidence),
        "predictor_reuse": _finite_rms(predictor, axis=(0, 1)),
        "emitted_token_anchor": _finite_rms(emitted, axis=(0, 1)),
    }


@dataclass(frozen=True)
class SourceBalancedECDF:
    """An empirical CDF in which every source group has equal total mass."""

    value: np.ndarray
    cumulative: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, sources: np.ndarray) -> "SourceBalancedECDF":
        values = np.asarray(values, dtype=np.float64)
        sources = np.asarray(sources).astype(str, copy=False)
        finite = np.isfinite(values)
        values, sources = values[finite], sources[finite]
        if not len(values):
            raise ValueError("an empirical CDF needs at least one finite value")
        _, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
        weight = 1.0 / counts[inverse]
        order = np.argsort(values, kind="stable")
        ordered_value = values[order]
        ordered_weight = weight[order]
        unique, first, multiplicity = np.unique(
            ordered_value, return_index=True, return_counts=True
        )
        cumulative = np.cumsum(ordered_weight)
        right = first + multiplicity - 1
        return cls(unique, cumulative[right] / cumulative[-1])

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        result = np.full(values.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(values)
        index = np.searchsorted(self.value, values[finite], side="right") - 1
        selected = np.zeros(index.shape, dtype=np.float64)
        inside = index >= 0
        selected[inside] = self.cumulative[index[inside]]
        result[finite] = selected
        return result


@dataclass(frozen=True)
class NuisanceBinner:
    """Train-fitted cells for position, entropy and target log-probability."""

    position_bins: int
    entropy_edges: np.ndarray
    logprob_edges: np.ndarray

    @staticmethod
    def _edges(values: np.ndarray, bins: int) -> np.ndarray:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        quantiles = np.arange(1, bins, dtype=np.float64) / bins
        return np.unique(np.quantile(finite, quantiles))

    @classmethod
    def fit(
        cls,
        relative_position: np.ndarray,
        entropy: np.ndarray,
        logprob: np.ndarray,
        *,
        position_bins: int = 8,
        state_bins: int = 4,
    ) -> "NuisanceBinner":
        return cls(
            position_bins,
            cls._edges(entropy, state_bins),
            cls._edges(logprob, state_bins),
        )

    def cells(
        self,
        relative_position: np.ndarray,
        entropy: np.ndarray,
        logprob: np.ndarray,
    ) -> np.ndarray:
        position = np.asarray(relative_position, dtype=np.float64)
        entropy = np.asarray(entropy, dtype=np.float64)
        logprob = np.asarray(logprob, dtype=np.float64)
        valid = np.isfinite(position) & np.isfinite(entropy) & np.isfinite(logprob)
        position_bin = np.minimum(
            np.floor(np.clip(position, 0.0, 1.0) * self.position_bins).astype(int),
            self.position_bins - 1,
        )
        entropy_bin = np.searchsorted(self.entropy_edges, entropy, side="right")
        logprob_bin = np.searchsorted(self.logprob_edges, logprob, side="right")
        cells = (
            (position_bin * (len(self.entropy_edges) + 1) + entropy_bin)
            * (len(self.logprob_edges) + 1)
            + logprob_bin
        )
        return np.where(valid, cells, -1)


class ConditionalECDF:
    """Source-balanced ECDFs within train-fitted nuisance cells."""

    def __init__(
        self,
        global_table: SourceBalancedECDF,
        tables: Mapping[int, SourceBalancedECDF],
    ) -> None:
        self.global_table = global_table
        self.tables = dict(tables)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        sources: np.ndarray,
        cells: np.ndarray,
        *,
        min_cell_values: int = 64,
        min_cell_sources: int = 3,
    ) -> "ConditionalECDF":
        values = np.asarray(values, dtype=np.float64)
        sources = np.asarray(sources).astype(str, copy=False)
        cells = np.asarray(cells, dtype=np.int64)
        global_table = SourceBalancedECDF.fit(values, sources)
        tables = {}
        for cell in np.unique(cells[cells >= 0]):
            selected = (cells == cell) & np.isfinite(values)
            if (
                selected.sum() >= min_cell_values
                and np.unique(sources[selected]).size >= min_cell_sources
            ):
                tables[int(cell)] = SourceBalancedECDF.fit(
                    values[selected], sources[selected]
                )
        return cls(global_table, tables)

    def score(self, values: np.ndarray, cells: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.int64)
        result = self.global_table.score(values)
        for cell in np.unique(cells):
            table = self.tables.get(int(cell))
            if table is not None:
                selected = cells == cell
                result[selected] = table.score(values[selected])
        return result


def compose_scores(tail: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compose registered AND/OR failure modes without learned weights."""

    transport_gap = _finite_extreme(
        [tail["route_demand"], tail["evidence_entry_deficit"]], "min"
    )
    entry = _finite_extreme(
        [transport_gap, tail["context_opposition"]], "min"
    )
    candidate_rejection = _finite_extreme(
        [tail["adoption_deficit"], tail["context_target_log_rank"]], "max"
    )
    adoption = _finite_extreme(
        [tail["context_distribution_js"], candidate_rejection], "min"
    )
    self_anchor = _finite_extreme(
        [tail["predictor_reuse"], tail["emitted_token_anchor"]], "max"
    )
    unsupported = _finite_extreme(
        [tail["context_opposition"], tail["adoption_deficit"]], "max"
    )
    override = _finite_extreme(
        [tail["late_evidence_route_loss"], self_anchor, unsupported], "min"
    )
    online = _finite_extreme([entry, adoption], "max")
    return {
        "entry_failure": entry,
        "adoption_failure": adoption,
        "override_candidate": override,
        "online_failure": online,
        "offline_failure": _finite_extreme([online, override], "max"),
    }


@dataclass(frozen=True)
class DetectorScores:
    raw: dict[str, np.ndarray]
    tail: dict[str, np.ndarray]
    score: dict[str, np.ndarray]


class ReanchorFailureDetector:
    """Task-specific unlabeled train calibration and frozen token scoring."""

    def __init__(
        self,
        binner: NuisanceBinner,
        calibrators: Mapping[str, ConditionalECDF],
    ) -> None:
        self.binner = binner
        self.calibrators = dict(calibrators)

    @classmethod
    def fit(
        cls,
        records: Sequence[Mapping[str, np.ndarray]],
        source_ids: Sequence[str],
    ) -> "ReanchorFailureDetector":
        if len(records) != len(source_ids) or not records:
            raise ValueError("calibration records and source IDs must align")
        combined = {
            name: np.concatenate([np.asarray(row[name]) for row in records])
            for name in (
                "relative_position",
                "baseline_entropy",
                "baseline_target_logprob",
                *RAW_FEATURES,
            )
        }
        sources = np.concatenate(
            [
                np.repeat(str(source), len(row["relative_position"]))
                for row, source in zip(records, source_ids, strict=True)
            ]
        )
        binner = NuisanceBinner.fit(
            combined["relative_position"],
            combined["baseline_entropy"],
            combined["baseline_target_logprob"],
        )
        cells = binner.cells(
            combined["relative_position"],
            combined["baseline_entropy"],
            combined["baseline_target_logprob"],
        )
        calibrators = {
            name: ConditionalECDF.fit(combined[name], sources, cells)
            for name in RAW_FEATURES
        }
        return cls(binner, calibrators)

    def score(self, raw: Mapping[str, np.ndarray]) -> DetectorScores:
        cells = self.binner.cells(
            raw["relative_position"],
            raw["baseline_entropy"],
            raw["baseline_target_logprob"],
        )
        tail = {
            name: self.calibrators[name].score(raw[name], cells)
            for name in RAW_FEATURES
        }
        return DetectorScores(dict(raw), tail, compose_scores(tail))
