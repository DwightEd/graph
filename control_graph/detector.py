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
    contributions: dict[str, float]


class GraphAnomalyDetector:
    """Fit a robust normal profile and score graph-edge deviations."""

    def __init__(self) -> None:
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, graphs: list[ControlGraph]) -> GraphAnomalyDetector:
        if len(graphs) < 4:
            raise ValueError("graph anomaly fit requires at least four graphs")
        values = _matrix(graphs)
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        iqr = (np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)) / 1.349
        std = np.std(values, axis=0)
        floor = np.maximum(np.abs(center) * 0.05, 0.05)
        self.center = center
        self.scale = np.where(mad > 1e-8, mad, np.where(iqr > 1e-8, iqr, np.maximum(std, floor)))
        return self

    def score(self, graphs: list[ControlGraph]) -> tuple[AnomalyScore, ...]:
        if self.center is None or self.scale is None:
            raise RuntimeError("fit must be called before score")
        standardized = (_matrix(graphs) - self.center) / self.scale
        squared = standardized**2
        scores = []
        for graph, row in zip(graphs, squared, strict=True):
            contributions = {
                name: float(value) for name, value in zip(EDGE_ORDER, row, strict=True)
            }
            dominant = max(contributions, key=contributions.__getitem__)
            scores.append(
                AnomalyScore(
                    event_id=graph.event_id,
                    source_id=graph.source_id,
                    split=graph.split,
                    relation=graph.relation,
                    score=float(np.mean(row)),
                    dominant_edge=dominant,
                    contributions=contributions,
                )
            )
        return tuple(scores)


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
