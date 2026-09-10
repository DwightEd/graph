"""File workflows for graph construction and label-free anomaly scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from control_graph.data import load_factorial_events
from control_graph.detector import GraphAnomalyDetector
from control_graph.graph import EDGE_ORDER, ControlGraph, ControlGraphBuilder


@dataclass(frozen=True)
class BuildConfig:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class DetectionConfig:
    graph_path: Path
    output_path: Path
    fit_split: str = "train"
    score_split: str = "test"


class BuildGraphDataset:
    """Build one canonical graph per factorial commitment event."""

    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def run(self) -> dict:
        if self.config.output_path.exists():
            raise FileExistsError(f"graph output already exists: {self.config.output_path}")
        events = load_factorial_events(self.config.input_path)
        builder = ControlGraphBuilder()
        graphs = [builder.build(event) for event in events]
        _write_jsonl(self.config.output_path, [graph.to_record() for graph in graphs])
        return {
            "schema": "control-graph/build-summary@1",
            "graphs": len(graphs),
            "sources": len({graph.source_id for graph in graphs}),
            "splits": sorted({graph.split for graph in graphs}),
            "input_sha256": _sha256(self.config.input_path),
            "output": str(self.config.output_path),
        }


class DetectGraphAnomalies:
    """Fit on one unlabeled split and score a source-disjoint split."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    def run(self) -> dict:
        output = self.config.output_path
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"detection output directory is not empty: {output}")
        graphs = _load_graphs(self.config.graph_path)
        fit_graphs = [graph for graph in graphs if graph.split == self.config.fit_split]
        score_graphs = [graph for graph in graphs if graph.split == self.config.score_split]
        if not score_graphs:
            raise ValueError(f"no graphs found for score split {self.config.score_split!r}")
        fit_sources = {graph.source_id for graph in fit_graphs}
        score_sources = {graph.source_id for graph in score_graphs}
        overlap = fit_sources.intersection(score_sources)
        if overlap:
            raise ValueError("fit and score splits must be source-disjoint")

        detector = GraphAnomalyDetector().fit(fit_graphs)
        scores = detector.score(score_graphs)
        output.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            output / "scores.jsonl",
            [
                {
                    "schema": "control-graph/anomaly-score@1",
                    "event_id": score.event_id,
                    "source_id": score.source_id,
                    "split": score.split,
                    "relation": score.relation,
                    "anomaly_score": score.score,
                    "dominant_edge": score.dominant_edge,
                    "contributions": score.contributions,
                }
                for score in scores
            ],
        )
        model = {
            "schema": "control-graph/robust-reference@1",
            "edge_order": list(EDGE_ORDER),
            "profiles": {
                relation: {
                    "center": profile.center.tolist(),
                    "scale": profile.scale.tolist(),
                }
                for relation, profile in detector.profiles.items()
            },
            "fit_split": self.config.fit_split,
            "fit_sources": sorted(fit_sources),
            "graph_sha256": _sha256(self.config.graph_path),
        }
        (output / "model.json").write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = {
            "schema": "control-graph/detection-summary@1",
            "fit_graphs": len(fit_graphs),
            "fit_sources": len(fit_sources),
            "scored_graphs": len(score_graphs),
            "scored_sources": len(score_sources),
            "fit_split": self.config.fit_split,
            "score_split": self.config.score_split,
            "labels_used": False,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary


def _load_graphs(path: Path) -> list[ControlGraph]:
    if not path.is_file():
        raise FileNotFoundError(f"control graph JSONL does not exist: {path}")
    graphs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            graphs.append(ControlGraph.from_record(json.loads(line)))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid graph at {path}:{line_number}: {error}") from error
    if not graphs:
        raise ValueError(f"control graph JSONL is empty: {path}")
    if len({graph.event_id for graph in graphs}) != len(graphs):
        raise ValueError("control graph JSONL contains duplicate event IDs")
    return graphs


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
