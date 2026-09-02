"""Label-free conditional innovation on complete register-graph sequences."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .graph import GraphSequence
from .metric import BLOCK_NAMES, RouteMetric, as_array

HISTORY = 2
POSITION_BINS = 10


def nearest_condition(
    available: Sequence[tuple[int, int]], requested: tuple[int, int]
) -> tuple[int, int]:
    """Use the closest populated position/length cell in finite references."""

    return min(
        available,
        key=lambda key: (
            abs(key[0] - requested[0]) + abs(key[1] - requested[1]),
            abs(key[0] - requested[0]),
            key,
        ),
    )


@dataclass(frozen=True)
class GraphRecord:
    """One label-free graph sequence and its sampling coordinates."""

    source_id: str
    prompt_length: int
    sequence: GraphSequence


@dataclass(frozen=True)
class CandidateWindow:
    source_id: str
    prompt_length: int
    sequence: GraphSequence
    end: int
    position_bin: int
    length_bin: int


@dataclass(frozen=True)
class PrototypeBank:
    """Actual observed windows retained as a small multimodal reference bank."""

    tensor: dict[str, np.ndarray]
    weight: np.ndarray
    bandwidth: float

    @property
    def count(self) -> int:
        return len(self.weight)


def response_position_bin(sequence: GraphSequence, token: int) -> int:
    position = as_array(sequence.prediction_position)
    relative = (float(position[token] - position[0]) + 0.5) / len(position)
    return min(int(relative * POSITION_BINS), POSITION_BINS - 1)


def prompt_length_edges(records: Sequence[GraphRecord]) -> np.ndarray:
    """Quartiles use one prompt length per source, so prolific sources do not win."""

    by_source: dict[str, int] = {}
    for record in sorted(records, key=lambda item: item.source_id):
        by_source.setdefault(record.source_id, int(record.prompt_length))
    if not by_source:
        raise ValueError("at least one reference source is required")
    return np.quantile(
        np.asarray(tuple(by_source.values()), dtype=np.float64),
        (0.25, 0.5, 0.75),
    )


def prompt_length_bin(prompt_length: int, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, prompt_length, side="right"))


def valid_window(sequence: GraphSequence, end: int) -> bool:
    valid = as_array(sequence.valid)
    return end >= HISTORY and bool(valid[end])


def candidate_windows(
    records: Sequence[GraphRecord], length_edges: np.ndarray
) -> list[CandidateWindow]:
    """Choose one valid token per source and response-position decile."""

    selected: dict[tuple[str, int], tuple[float, CandidateWindow]] = {}
    for record in records:
        sequence = record.sequence
        for token in range(HISTORY, len(sequence.valid)):
            if not valid_window(sequence, token):
                continue
            position_bin = response_position_bin(sequence, token)
            position = as_array(sequence.prediction_position)
            relative = (float(position[token] - position[0]) + 0.5) / len(
                sequence.valid
            )
            candidate = CandidateWindow(
                source_id=record.source_id,
                prompt_length=record.prompt_length,
                sequence=sequence,
                end=token,
                position_bin=position_bin,
                length_bin=prompt_length_bin(record.prompt_length, length_edges),
            )
            error = abs(relative - (position_bin + 0.5) / POSITION_BINS)
            key = (record.source_id, position_bin)
            previous = selected.get(key)
            if previous is None or error < previous[0]:
                selected[key] = (error, candidate)
    return [value[1] for _, value in sorted(selected.items(), key=lambda item: item[0])]


def frame_pairs(
    candidates: Sequence[CandidateWindow],
) -> list[tuple[tuple[GraphSequence, int], tuple[GraphSequence, int]]]:
    """Deterministic cross-source pairs used only to normalize metric blocks."""

    groups: dict[tuple[int, int], list[CandidateWindow]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate.position_bin, candidate.length_bin)].append(candidate)

    pairs = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.source_id)
        pairs.extend(
            ((left.sequence, left.end), (right.sequence, right.end))
            for left, right in zip(ordered[::2], ordered[1::2], strict=False)
        )
    return pairs


def window_distance(
    metric: RouteMetric, left: CandidateWindow, right: CandidateWindow
) -> float:
    context = np.mean(
        [
            metric.distance(
                (left.sequence, left.end - lag),
                (right.sequence, right.end - lag),
            )
            for lag in range(HISTORY, 0, -1)
        ]
    )
    current = metric.distance((left.sequence, left.end), (right.sequence, right.end))
    return float(context + current)


def window_batch(candidates: Sequence[CandidateWindow]) -> dict[str, np.ndarray]:
    return {
        name: np.stack(
            [
                as_array(getattr(item.sequence, name))[
                    item.end - HISTORY : item.end + 1
                ]
                for item in candidates
            ]
        )
        for name in BLOCK_NAMES
    }


def window_distances_to_batch(
    metric: RouteMetric,
    candidate: CandidateWindow,
    batch: dict[str, np.ndarray],
) -> np.ndarray:
    distance = np.zeros(len(next(iter(batch.values()))), dtype=np.float64)
    for step in range(HISTORY):
        distance += (
            metric.distances_to_batch(
                candidate.sequence,
                candidate.end - HISTORY + step,
                {name: value[:, step] for name, value in batch.items()},
            )
            / HISTORY
        )
    distance += metric.distances_to_batch(
        candidate.sequence,
        candidate.end,
        {name: value[:, HISTORY] for name, value in batch.items()},
    )
    return distance


def select_prototypes(
    candidates: Sequence[CandidateWindow],
    metric: RouteMetric,
    count: int,
) -> tuple[CandidateWindow, ...]:
    """Deterministic farthest-first coreset of actual observed windows."""

    ordered = sorted(candidates, key=lambda item: (item.source_id, item.end))
    batch = window_batch(ordered)
    selected = [ordered[0]]
    nearest = window_distances_to_batch(metric, selected[0], batch)
    while len(selected) < min(count, len(ordered)):
        next_index = int(np.argmax(nearest))
        if nearest[next_index] <= 1e-12:
            break
        selected.append(ordered[next_index])
        distance = window_distances_to_batch(metric, selected[-1], batch)
        nearest = np.minimum(nearest, distance)
    return tuple(selected)


def prototype_bank(
    candidates: Sequence[CandidateWindow],
    metric: RouteMetric,
    count: int,
) -> PrototypeBank:
    prototypes = select_prototypes(candidates, metric, count)
    tensor = window_batch(prototypes)
    candidates_batch = window_batch(candidates)
    distance = np.stack(
        [
            window_distances_to_batch(metric, prototype, candidates_batch)
            for prototype in prototypes
        ]
    )
    assignment = np.argmin(distance, axis=0)
    assigned_distance = distance[assignment, np.arange(len(candidates))]
    count_by_prototype = np.bincount(assignment, minlength=len(prototypes)).astype(
        np.float64
    )
    weight = count_by_prototype / count_by_prototype.sum()

    positive = np.asarray(assigned_distance)
    positive = positive[positive > 1e-12]
    if not len(positive):
        adjacent = [
            window_distance(metric, left, right) for left, right in pairwise(prototypes)
        ]
        positive = np.asarray([value for value in adjacent if value > 1e-12])
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    return PrototypeBank(tensor=tensor, weight=weight, bandwidth=bandwidth)


def row_logsumexp(value: np.ndarray) -> np.ndarray:
    maximum = np.max(value, axis=1)
    return maximum + np.log(np.exp(value - maximum[:, None]).sum(axis=1))


@dataclass(frozen=True)
class SourceECDF:
    """Source-equal empirical calibration tables for raw innovation energy."""

    table: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    length_edges: np.ndarray

    @classmethod
    def fit(
        cls,
        records: Sequence[GraphRecord],
        scores: Sequence[np.ndarray],
        length_edges: np.ndarray,
    ) -> SourceECDF:
        grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record, score in zip(records, scores, strict=True):
            for token, value in enumerate(np.asarray(score, dtype=np.float64)):
                if not np.isfinite(value):
                    continue
                key = (
                    response_position_bin(record.sequence, token),
                    prompt_length_bin(record.prompt_length, length_edges),
                )
                grouped[key][record.source_id].append(float(value))

        tables = {}
        for key, sources in grouped.items():
            value = []
            weight = []
            for source_values in sources.values():
                value.extend(source_values)
                weight.extend([1.0 / len(source_values)] * len(source_values))
            value_array = np.asarray(value, dtype=np.float64)
            weight_array = np.asarray(weight, dtype=np.float64)
            order = np.argsort(value_array, kind="stable")
            value_array = value_array[order]
            cumulative = np.cumsum(weight_array[order]) / weight_array.sum()
            tables[key] = (value_array, cumulative)
        if not tables:
            raise ValueError("calibration records contain no finite scores")
        return cls(tables, np.asarray(length_edges, dtype=np.float64))

    def transform(self, record: GraphRecord, score: np.ndarray) -> np.ndarray:
        calibrated = np.full(len(score), np.nan, dtype=np.float32)
        available = tuple(self.table)
        for token, value in enumerate(np.asarray(score, dtype=np.float64)):
            if not np.isfinite(value):
                continue
            key = (
                response_position_bin(record.sequence, token),
                prompt_length_bin(record.prompt_length, self.length_edges),
            )
            reference, cumulative = self.table[nearest_condition(available, key)]
            location = np.searchsorted(reference, value, side="right")
            calibrated[token] = cumulative[location - 1] if location else 0.0
        return calibrated


class TransitionDetector:
    """Conditional kernel energy over full, multimodal graph-state windows."""

    def __init__(self, prototype_count: int = 8) -> None:
        self.prototype_count = max(1, min(int(prototype_count), 8))
        self.metric: RouteMetric | None = None
        self.length_edges: np.ndarray | None = None
        self.reference: dict[tuple[int, int], PrototypeBank] = {}
        self.reference_sources: frozenset[str] = frozenset()
        self.calibrator: SourceECDF | None = None
        self.independent_calibrator: SourceECDF | None = None

    def fit(self, records: Sequence[GraphRecord]) -> TransitionDetector:
        records = tuple(records)
        if not records:
            raise ValueError("at least one reference record is required")
        self.reference_sources = frozenset(record.source_id for record in records)
        self.length_edges = prompt_length_edges(records)
        candidates = candidate_windows(records, self.length_edges)
        if not candidates:
            raise ValueError("reference records contain no valid two-step windows")
        self.metric = RouteMetric.fit(frame_pairs(candidates))

        grouped: dict[tuple[int, int], list[CandidateWindow]] = defaultdict(list)
        for candidate in candidates:
            grouped[(candidate.position_bin, candidate.length_bin)].append(candidate)
        self.reference = {
            key: prototype_bank(group, self.metric, self.prototype_count)
            for key, group in grouped.items()
        }
        return self

    def raw_scores(self, record: GraphRecord) -> tuple[np.ndarray, np.ndarray]:
        metric, length_edges = self._fitted()
        sequence = record.sequence
        conditional = np.full(len(sequence.valid), np.nan, dtype=np.float32)
        independent = np.full(len(sequence.valid), np.nan, dtype=np.float32)

        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for token in range(HISTORY, len(sequence.valid)):
            if not valid_window(sequence, token):
                continue
            key = (
                response_position_bin(sequence, token),
                prompt_length_bin(record.prompt_length, length_edges),
            )
            grouped[key].append(token)

        available = tuple(self.reference)
        for key, token_list in grouped.items():
            bank = self.reference[nearest_condition(available, key)]
            token = np.asarray(token_list, dtype=np.int64)
            context = np.zeros((len(token), bank.count), dtype=np.float64)
            for step in range(HISTORY):
                prototype = {
                    name: value[:, step] for name, value in bank.tensor.items()
                }
                context += (
                    metric.distances_from_indices_to_batch(
                        sequence, token - HISTORY + step, prototype
                    )
                    / HISTORY
                )
            current = metric.distances_from_indices_to_batch(
                sequence,
                token,
                {name: value[:, HISTORY] for name, value in bank.tensor.items()},
            )
            log_weight = np.log(bank.weight)[None]
            past_energy = row_logsumexp(log_weight - context / bank.bandwidth)
            joint_energy = row_logsumexp(
                log_weight - (context + current) / bank.bandwidth
            )
            conditional[token] = past_energy - joint_energy
            independent[token] = -row_logsumexp(log_weight - current / bank.bandwidth)
        return conditional, independent

    def raw_score(self, record: GraphRecord) -> np.ndarray:
        return self.raw_scores(record)[0]

    def calibrate(self, records: Sequence[GraphRecord]) -> TransitionDetector:
        _, length_edges = self._fitted()
        records = tuple(records)
        overlap = self.reference_sources.intersection(
            record.source_id for record in records
        )
        if overlap:
            raise ValueError("reference and calibration sources must be disjoint")
        raw = [self.raw_scores(record) for record in records]
        self.calibrator = SourceECDF.fit(
            records, [primary for primary, _ in raw], length_edges
        )
        self.independent_calibrator = SourceECDF.fit(
            records, [control for _, control in raw], length_edges
        )
        return self

    def score(self, record: GraphRecord) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("calibrate the detector before scoring")
        return self.calibrator.transform(record, self.raw_score(record))

    def independent_score(self, record: GraphRecord) -> np.ndarray:
        if self.independent_calibrator is None:
            raise RuntimeError("calibrate the detector before scoring")
        return self.independent_calibrator.transform(record, self.raw_scores(record)[1])

    def save(self, path: str | Path) -> None:
        metric, length_edges = self._fitted()
        keys = sorted(self.reference)
        starts = [0]
        prototype_weight = []
        block = {name: [] for name in BLOCK_NAMES}
        bandwidth = []
        for key in keys:
            bank = self.reference[key]
            starts.append(starts[-1] + bank.count)
            prototype_weight.append(bank.weight)
            bandwidth.append(bank.bandwidth)
            for name in BLOCK_NAMES:
                block[name].append(bank.tensor[name])

        arrays = {
            **metric.arrays(),
            "prototype_count": np.asarray(self.prototype_count, dtype=np.int32),
            "length_edges": length_edges,
            "reference_source": np.asarray(sorted(self.reference_sources)),
            "reference_key": np.asarray(keys, dtype=np.int8),
            "reference_start": np.asarray(starts, dtype=np.int32),
            "reference_bandwidth": np.asarray(bandwidth, dtype=np.float64),
            "prototype_weight": np.concatenate(prototype_weight),
            **{
                f"prototype_{name}": np.concatenate(values)
                for name, values in block.items()
            },
        }
        if self.calibrator is not None:
            arrays.update(self._calibration_arrays("primary", self.calibrator))
        if self.independent_calibrator is not None:
            arrays.update(
                self._calibration_arrays("independent", self.independent_calibrator)
            )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> TransitionDetector:
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        detector = cls(int(arrays["prototype_count"]))
        detector.metric = RouteMetric.from_arrays(arrays)
        detector.length_edges = np.asarray(arrays["length_edges"], dtype=np.float64)
        detector.reference_sources = frozenset(
            str(value) for value in arrays["reference_source"]
        )
        keys = [tuple(map(int, key)) for key in arrays["reference_key"]]
        starts = arrays["reference_start"]
        detector.reference = {}
        for number, key in enumerate(keys):
            location = slice(int(starts[number]), int(starts[number + 1]))
            detector.reference[key] = PrototypeBank(
                tensor={
                    name: arrays[f"prototype_{name}"][location] for name in BLOCK_NAMES
                },
                weight=arrays["prototype_weight"][location],
                bandwidth=float(arrays["reference_bandwidth"][number]),
            )
        if "primary_calibration_key" in arrays:
            detector.calibrator = detector._load_calibrator(arrays, "primary")
        if "independent_calibration_key" in arrays:
            detector.independent_calibrator = detector._load_calibrator(
                arrays, "independent"
            )
        return detector

    def _fitted(self) -> tuple[RouteMetric, np.ndarray]:
        if self.metric is None or self.length_edges is None or not self.reference:
            raise RuntimeError("fit the detector before scoring")
        return self.metric, self.length_edges

    @staticmethod
    def _calibration_arrays(
        prefix: str, calibrator: SourceECDF
    ) -> dict[str, np.ndarray]:
        keys = sorted(calibrator.table)
        starts = [0]
        value = []
        cumulative = []
        for key in keys:
            current_value, current_cumulative = calibrator.table[key]
            starts.append(starts[-1] + len(current_value))
            value.append(current_value)
            cumulative.append(current_cumulative)
        return {
            f"{prefix}_calibration_key": np.asarray(keys, dtype=np.int8),
            f"{prefix}_calibration_start": np.asarray(starts, dtype=np.int32),
            f"{prefix}_calibration_value": np.concatenate(value),
            f"{prefix}_calibration_cumulative": np.concatenate(cumulative),
        }

    def _load_calibrator(
        self, arrays: dict[str, np.ndarray], prefix: str
    ) -> SourceECDF:
        keys = [tuple(map(int, key)) for key in arrays[f"{prefix}_calibration_key"]]
        starts = arrays[f"{prefix}_calibration_start"]
        table = {}
        for number, key in enumerate(keys):
            location = slice(int(starts[number]), int(starts[number + 1]))
            table[key] = (
                arrays[f"{prefix}_calibration_value"][location],
                arrays[f"{prefix}_calibration_cumulative"][location],
            )
        return SourceECDF(table, np.asarray(self.length_edges, dtype=np.float64))
