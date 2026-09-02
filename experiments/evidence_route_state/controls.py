"""Equation-locked prompt-route-collapse control from the earlier QA audit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RouteCollapseControl:
    """Source-equal nuisance fit and held-out position calibration."""

    coefficients: np.ndarray
    scales: np.ndarray
    position_tables: tuple[tuple[np.ndarray, np.ndarray], ...]

    @classmethod
    def fit(
        cls,
        nuisance_records: Sequence[Mapping[str, object]],
        calibration_records: Sequence[Mapping[str, object]],
    ) -> RouteCollapseControl:
        """Fit position/length WLS and a source-disjoint lower-volume ECDF."""

        nuisance = tuple(nuisance_records)
        source_counts = _source_counts(nuisance)
        rows = [
            (
                _design(len(record["volume"]), int(record["prompt_length"])),
                np.asarray(record["volume"], dtype=np.float64),
                np.full(
                    len(record["volume"]),
                    1.0
                    / source_counts[str(record["source_id"])]
                    / len(record["volume"]),
                ),
            )
            for record in nuisance
        ]
        design = np.concatenate([row[0] for row in rows])
        weight = np.concatenate([row[2] for row in rows])
        root_weight = np.sqrt(weight)
        layers = rows[0][1].shape[1]
        coefficients = np.empty((layers, design.shape[1]), dtype=np.float64)
        scales = np.empty(layers, dtype=np.float64)

        for layer in range(layers):
            value = np.concatenate([row[1][:, layer] for row in rows])
            coefficients[layer] = np.linalg.lstsq(
                design * root_weight[:, None],
                value * root_weight,
                rcond=None,
            )[0]
            residual = value - design @ coefficients[layer]
            center = _weighted_median(residual, weight)
            scales[layer] = max(
                1.4826 * _weighted_median(np.abs(residual - center), weight),
                1e-3,
            )

        calibration = tuple(calibration_records)
        raw = [
            _raw_score(
                np.asarray(record["volume"], dtype=np.float64),
                int(record["prompt_length"]),
                coefficients,
                scales,
            )
            for record in calibration
        ]
        tables = _position_tables(calibration, raw, _source_counts(calibration))
        return cls(coefficients, scales, tables)

    def raw_score(self, record: Mapping[str, object]) -> np.ndarray:
        """Return the positive lower-volume residual averaged over layers."""

        return _raw_score(
            np.asarray(record["volume"], dtype=np.float64),
            int(record["prompt_length"]),
            self.coefficients,
            self.scales,
        )

    def score(self, record: Mapping[str, object]) -> np.ndarray:
        """Map route contraction through the matching response-position ECDF."""

        value = self.raw_score(record)
        score = np.empty(len(value), dtype=np.float32)
        for token, bucket in enumerate(_position_bucket(len(value))):
            reference, cumulative = self.position_tables[int(bucket)]
            location = np.searchsorted(reference, value[token], side="right")
            score[token] = cumulative[location - 1] if location else 0.0
        return score

    def save(self, path: str | Path) -> None:
        arrays = {"coefficients": self.coefficients, "scales": self.scales}
        for bucket, (value, cumulative) in enumerate(self.position_tables):
            arrays[f"value_{bucket}"] = value
            arrays[f"cumulative_{bucket}"] = cumulative
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> RouteCollapseControl:
        with np.load(path) as stored:
            tables = tuple(
                (stored[f"value_{bucket}"], stored[f"cumulative_{bucket}"])
                for bucket in range(10)
            )
            return cls(stored["coefficients"], stored["scales"], tables)


def prompt_log_volume(
    effective_sources: np.ndarray,
    effective_rank: np.ndarray,
    anchor_source: np.ndarray,
    window: int = 4,
) -> np.ndarray:
    """Return the locked log route volume with shape ``[token, layer]``."""

    sources = np.asarray(effective_sources, dtype=np.float64)
    rank = np.asarray(effective_rank, dtype=np.float64)
    anchors = _temporal_anchor_support(anchor_source, window)
    volume = (
        np.log(np.maximum(sources, 1.0))
        + np.log(np.maximum(rank, 1.0))
        + np.log(np.maximum(anchors, 1.0))
    )
    return volume.T


def _temporal_anchor_support(anchor_source: np.ndarray, window: int) -> np.ndarray:
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
                support[layer, token] = np.exp(
                    -(probability * np.log(probability)).sum()
                )
    return support


def _source_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["source_id"])] += 1
    return counts


def _design(tokens: int, prompt_length: int) -> np.ndarray:
    position = (np.arange(tokens, dtype=np.float64) + 0.5) / max(tokens, 1)
    length = np.full(tokens, np.log1p(prompt_length + tokens))
    return np.column_stack((np.ones(tokens), position, np.square(position), length))


def _weighted_median(value: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    middle = np.searchsorted(cumulative, cumulative[-1] / 2)
    return float(value[order[middle]])


def _raw_score(
    volume: np.ndarray,
    prompt_length: int,
    coefficients: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    expected = _design(len(volume), prompt_length) @ coefficients.T
    standardized = (expected - volume) / scales[None]
    return np.maximum(standardized, 0.0).mean(axis=1)


def _position_bucket(tokens: int) -> np.ndarray:
    return np.minimum((np.arange(tokens) * 10) // max(tokens, 1), 9)


def _position_tables(
    records: Sequence[Mapping[str, object]],
    values: Sequence[np.ndarray],
    source_counts: Mapping[str, int],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    pieces: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(10)]
    pooled_weights = []
    for record, value in zip(records, values, strict=True):
        token_weight = 1.0 / source_counts[str(record["source_id"])] / len(value)
        weights = np.full(len(value), token_weight)
        pooled_weights.append(weights)
        bucket = _position_bucket(len(value))
        for index in range(10):
            selected = bucket == index
            if selected.any():
                pieces[index].append((value[selected], weights[selected]))

    pooled = _weighted_ecdf(np.concatenate(values), np.concatenate(pooled_weights))
    return tuple(
        _weighted_ecdf(
            np.concatenate([piece[0] for piece in bucket]),
            np.concatenate([piece[1] for piece in bucket]),
        )
        if bucket
        else pooled
        for bucket in pieces
    )


def _weighted_ecdf(
    value: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order])
    return value[order], cumulative / cumulative[-1]
