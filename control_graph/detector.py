"""Robust one-class scoring for fixed-role control graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control_graph.graph import EDGE_ORDER, ControlGraph


@dataclass(frozen=True)
class AnomalyScore:
    event_id: str
    source_id: str
    split: str
    relation: str
    score: float
    dominant_edge: str
    edge_deviations: dict[str, float]


@dataclass(frozen=True)
class ReferenceProfile:
    center: np.ndarray
    scale: np.ndarray


class GraphAnomalyDetector:
    """Fit robust relation-conditional profiles and score edge deviations."""

    def __init__(self) -> None:
        self.profiles: dict[str, ReferenceProfile] = {}

    def fit(self, graphs: list[ControlGraph]) -> GraphAnomalyDetector:
        relations = sorted({graph.relation for graph in graphs})
        if not relations:
            raise ValueError("at least one graph is required")
        self.profiles = {
            relation: _fit_profile(
                [graph for graph in graphs if graph.relation == relation], relation
            )
            for relation in relations
        }
        return self

    def score(self, graphs: list[ControlGraph]) -> tuple[AnomalyScore, ...]:
        if not self.profiles:
            raise RuntimeError("fit must be called before score")
        scores = []
        for graph in graphs:
            profile = self.profiles.get(graph.relation)
            if profile is None:
                raise ValueError(
                    f"no fitted reference profile for relation {graph.relation!r}"
                )
            row = (_matrix([graph])[0] - profile.center) / profile.scale
            deviations = {
                name: float(value) for name, value in zip(EDGE_ORDER, row, strict=True)
            }
            dominant = max(deviations, key=lambda name: abs(deviations[name]))
            scores.append(
                AnomalyScore(
                    event_id=graph.event_id,
                    source_id=graph.source_id,
                    split=graph.split,
                    relation=graph.relation,
                    score=float(np.mean(row**2)),
                    dominant_edge=dominant,
                    edge_deviations=deviations,
                )
            )
        return tuple(scores)


def _fit_profile(graphs: list[ControlGraph], relation: str) -> ReferenceProfile:
    if len(graphs) < 4:
        raise ValueError(
            f"graph anomaly fit requires at least four graphs for relation {relation!r}"
        )
    values = _matrix(graphs)
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center), axis=0)
    iqr = (np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)) / 1.349
    std = np.std(values, axis=0)
    floor = np.maximum(np.abs(center) * 0.05, 0.05)
    scale = np.where(mad > 1e-8, mad, np.where(iqr > 1e-8, iqr, np.maximum(std, floor)))
    return ReferenceProfile(center=center, scale=scale)


def _matrix(graphs: list[ControlGraph]) -> np.ndarray:
    if not graphs:
        raise ValueError("at least one graph is required")
    rows = []
    for graph in graphs:
        signature = graph.signature()
        if tuple(signature) != EDGE_ORDER:
            raise ValueError("control graph has a non-canonical edge schema")
        rows.append([signature[name] for name in EDGE_ORDER])
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("control graph edge weights must be finite")
    return values
