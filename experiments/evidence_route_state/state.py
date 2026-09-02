"""Mechanism-fixed observations derived from evidence-route lineage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .lineage import RouteLineage

EPSILON = 1e-12


@dataclass(frozen=True)
class RouteVolume:
    """The three degrees of freedom behind the earlier route-collapse result."""

    effective_sources: np.ndarray
    effective_head_rank: np.ndarray
    effective_anchors: np.ndarray
    log_volume: np.ndarray
    normalized: np.ndarray


@dataclass(frozen=True)
class RouteState:
    """Per-token lineage geometry captured before task-level calibration."""

    query_position: np.ndarray
    prediction_position: np.ndarray
    volume: RouteVolume
    raw_contraction: np.ndarray
    takeover: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class EquationLockedRouteCollapseControl:
    """Label-free control that locks the f7344e2 equation and score direction.

    This is not a numerical replay of that commit: its nuisance model and ECDF
    are fitted on source-disjoint subsets supplied by the current physical
    train/test experiment.
    """

    coefficients: np.ndarray
    scales: np.ndarray
    position_tables: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def fit(
        cls,
        nuisance_records: Sequence[Mapping[str, object]],
        calibration_records: Sequence[Mapping[str, object]],
    ) -> EquationLockedRouteCollapseControl:
        """Fit nuisance WLS/MAD, then a source-disjoint position ECDF."""

        records = tuple(nuisance_records)
        source_counts: dict[str, int] = defaultdict(int)
        for record in records:
            source_counts[str(record["source_id"])] += 1

        rows = []
        for record in records:
            volume = np.asarray(record["volume"], dtype=np.float64)
            source_weight = 1.0 / source_counts[str(record["source_id"])]
            rows.append(
                (
                    route_design(len(volume), int(record["prompt_length"])),
                    volume,
                    np.full(len(volume), source_weight / len(volume)),
                )
            )

        layers = rows[0][1].shape[1]
        coefficients = np.zeros((layers, 4), dtype=np.float64)
        scales = np.ones(layers, dtype=np.float64)
        design = np.concatenate([row[0] for row in rows])
        weight = np.concatenate([row[2] for row in rows])
        root_weight = np.sqrt(weight)
        for layer in range(layers):
            value = np.concatenate([row[1][:, layer] for row in rows])
            coefficients[layer] = np.linalg.lstsq(
                design * root_weight[:, None],
                value * root_weight,
                rcond=None,
            )[0]
            residual = value - design @ coefficients[layer]
            center = weighted_median(residual, weight)
            scales[layer] = max(
                1.4826 * weighted_median(np.abs(residual - center), weight),
                1e-3,
            )

        calibration_records = tuple(calibration_records)
        calibration_source_counts: dict[str, int] = defaultdict(int)
        for record in calibration_records:
            calibration_source_counts[str(record["source_id"])] += 1
        raw = [
            raw_route_collapse(
                np.asarray(record["volume"], dtype=np.float64),
                int(record["prompt_length"]),
                coefficients,
                scales,
            )
            for record in calibration_records
        ]
        tables = position_ecdf_tables(
            calibration_records,
            raw,
            calibration_source_counts,
        )
        return cls(coefficients, scales, tables)

    def raw_score(self, record: Mapping[str, object]) -> np.ndarray:
        return raw_route_collapse(
            np.asarray(record["volume"], dtype=np.float64),
            int(record["prompt_length"]),
            self.coefficients,
            self.scales,
        )

    def score(self, record: Mapping[str, object]) -> np.ndarray:
        """Map lower-than-expected volume through the held-out calibration ECDF."""

        value = self.raw_score(record)
        result = np.empty(len(value), dtype=np.float32)
        for index, bucket in enumerate(position_bucket(len(value))):
            reference, cumulative = self.position_tables[int(bucket)]
            location = np.searchsorted(reference, value[index], side="right")
            result[index] = cumulative[location - 1] if location else 0.0
        return result

    def save(self, path: str | Path) -> None:
        arrays = {
            "coefficients": self.coefficients,
            "scales": self.scales,
        }
        for index, (reference, cumulative) in enumerate(self.position_tables):
            arrays[f"reference_{index}"] = reference
            arrays[f"cumulative_{index}"] = cumulative
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> EquationLockedRouteCollapseControl:
        with np.load(path) as arrays:
            tables = tuple(
                (arrays[f"reference_{index}"], arrays[f"cumulative_{index}"])
                for index in range(10)
            )
            return cls(arrays["coefficients"], arrays["scales"], tables)


def prompt_log_volume(
    effective_sources: np.ndarray,
    effective_rank: np.ndarray,
    anchor_source: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """Locked f7344e2 prompt-route volume with shape ``[token, layer]``."""

    sources = np.asarray(effective_sources, dtype=np.float64)
    rank = np.asarray(effective_rank, dtype=np.float64)
    anchors = temporal_anchor_support(anchor_source, window=window)
    volume = (
        np.log(np.maximum(sources, 1.0))
        + np.log(np.maximum(rank, 1.0))
        + np.log(np.maximum(anchors, 1.0))
    )
    return volume.T


def route_volume(
    effective_sources: np.ndarray,
    effective_head_rank: np.ndarray,
    anchor_source: np.ndarray,
    query_position: np.ndarray,
    anchor_window: int = 4,
) -> RouteVolume:
    """Build source-token route volume from compact lineage topology fields."""

    sources = np.asarray(effective_sources, dtype=np.float64)
    rank = np.asarray(effective_head_rank, dtype=np.float64)
    anchors_by_head = np.asarray(anchor_source, dtype=np.int64)
    anchors = temporal_anchor_support(anchors_by_head, window=anchor_window)
    active = sources > EPSILON
    anchors[~active] = 0.0

    query = np.asarray(query_position, dtype=np.int64)
    source_limit = np.broadcast_to(np.maximum(query, 1), sources.shape)
    head_limit = np.full(sources.shape, anchors_by_head.shape[-1])
    visible_tokens = np.minimum(np.arange(sources.shape[1]) + 1, anchor_window)
    anchor_limit = np.minimum(source_limit, head_limit * visible_tokens)
    source_capacity = normalized_log_capacity(sources, source_limit, active)
    head_capacity = normalized_log_capacity(
        rank,
        np.minimum(head_limit, source_limit),
        active,
    )
    anchor_capacity = normalized_log_capacity(anchors, anchor_limit, active)
    log_volume = (
        np.log(np.maximum(sources, 1.0))
        + np.log(np.maximum(rank, 1.0))
        + np.log(np.maximum(anchors, 1.0))
    )

    return RouteVolume(
        effective_sources=sources.astype(np.float32),
        effective_head_rank=rank.astype(np.float32),
        effective_anchors=anchors.astype(np.float32),
        log_volume=log_volume.astype(np.float32),
        normalized=((source_capacity + head_capacity + anchor_capacity) / 3.0).astype(
            np.float32
        ),
    )


def build_route_state(
    lineage: RouteLineage,
    anchor_window: int = 4,
) -> RouteState:
    """Build raw label-free lineage geometry for later task-level calibration.

    Raw contraction is an attainable-capacity diagnostic. The detector instead
    calibrates log volume against position and length inside each training
    split. Takeover measures the unrooted share of known evidence/history flow.
    """

    volume = route_volume(
        to_numpy(lineage.effective_sources, np.float64),
        to_numpy(lineage.effective_head_rank, np.float64),
        to_numpy(lineage.anchor_source, np.int64),
        to_numpy(lineage.query_position, np.int64),
        anchor_window=anchor_window,
    )
    raw_contraction = 1.0 - volume.normalized.mean(axis=0)

    direct = to_numpy(lineage.prompt_evidence, np.float64)
    relay = to_numpy(lineage.grounded_response_relay, np.float64)
    unrooted = to_numpy(lineage.unrooted_response_feedback, np.float64)
    known_flow = (direct + relay + unrooted).sum(axis=(0, 2))
    unrooted_flow = unrooted.sum(axis=(0, 2))
    takeover = np.divide(
        unrooted_flow,
        known_flow,
        out=np.zeros_like(unrooted_flow),
        where=known_flow > EPSILON,
    )

    valid = to_numpy(lineage.history_valid, bool).copy()
    valid[:2] = False
    valid &= known_flow > EPSILON

    return RouteState(
        query_position=to_numpy(lineage.query_position, np.int64),
        prediction_position=to_numpy(lineage.prediction_position, np.int64),
        volume=volume,
        raw_contraction=raw_contraction.astype(np.float32),
        takeover=takeover.astype(np.float32),
        valid=valid,
    )


def route_observation(
    contraction: np.ndarray,
    takeover: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Combine calibrated contraction and takeover for valid token rows."""

    observation = np.full((len(contraction), 2), np.nan, dtype=np.float32)
    observation[valid, 0] = contraction[valid]
    observation[valid, 1] = takeover[valid]
    return observation


def entropy(probability: np.ndarray, axis: int | None = None) -> np.ndarray:
    positive = probability > 0
    terms = np.zeros_like(probability, dtype=np.float64)
    terms[positive] = probability[positive] * np.log(probability[positive])
    return -terms.sum(axis=axis)


def temporal_anchor_support(
    anchor_source: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """Effective prompt-anchor count across heads and recent response tokens."""

    anchors = np.asarray(anchor_source, dtype=np.int64)
    layers, tokens, _heads = anchors.shape
    support = np.ones((layers, tokens), dtype=np.float64)
    for layer in range(layers):
        for token in range(tokens):
            recent = anchors[layer, max(0, token - window + 1) : token + 1].ravel()
            recent = recent[recent >= 0]
            if len(recent):
                counts = np.unique(recent, return_counts=True)[1].astype(np.float64)
                probability = counts / counts.sum()
                support[layer, token] = np.exp(entropy(probability))
    return support


def route_design(tokens: int, prompt_length: int) -> np.ndarray:
    position = (np.arange(tokens, dtype=np.float64) + 0.5) / max(tokens, 1)
    length = np.full(tokens, np.log1p(prompt_length + tokens))
    return np.column_stack((np.ones(tokens), position, np.square(position), length))


def weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    middle = np.searchsorted(cumulative, cumulative[-1] / 2)
    return float(value[order[middle]])


def raw_route_collapse(
    volume: np.ndarray,
    prompt_length: int,
    coefficients: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    expected = route_design(len(volume), prompt_length) @ coefficients.T
    standardized = (expected - volume) / scales[None, :]
    return np.maximum(standardized, 0.0).mean(axis=1)


def position_bucket(tokens: int) -> np.ndarray:
    return np.minimum((np.arange(tokens) * 10) // max(tokens, 1), 9)


def position_ecdf_tables(
    records: Sequence[Mapping[str, object]],
    values: Sequence[np.ndarray],
    source_counts: Mapping[str, int],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    pieces: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(10)]
    pooled_weights = []
    for record, value in zip(records, values):
        source_weight = 1.0 / source_counts[str(record["source_id"])]
        token_weight = source_weight / len(value)
        pooled_weights.append(np.full(len(value), token_weight))
        buckets = position_bucket(len(value))
        for bucket in range(10):
            selected = value[buckets == bucket]
            if len(selected):
                pieces[bucket].append((selected, np.full(len(selected), token_weight)))

    def table(
        table_values: np.ndarray,
        table_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(table_values, kind="stable")
        cumulative = np.cumsum(table_weights[order])
        return table_values[order], cumulative / cumulative[-1]

    pooled = table(np.concatenate(values), np.concatenate(pooled_weights))
    tables = []
    for bucket in range(10):
        if not pieces[bucket]:
            tables.append(pooled)
            continue
        bucket_values = np.concatenate([piece[0] for piece in pieces[bucket]])
        bucket_weights = np.concatenate([piece[1] for piece in pieces[bucket]])
        tables.append(table(bucket_values, bucket_weights))
    return tuple(tables)


def normalized_log_capacity(
    effective: np.ndarray,
    attainable: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Map an effective count to [0, 1] relative to its attainable count."""

    result = np.zeros_like(effective, dtype=np.float64)
    informative = active & (attainable > 1)
    result[informative] = np.log(np.maximum(effective[informative], 1.0)) / np.log(
        attainable[informative]
    )
    result[active & (attainable <= 1)] = 1.0
    return np.clip(result, 0.0, 1.0)


def to_numpy(value: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)
