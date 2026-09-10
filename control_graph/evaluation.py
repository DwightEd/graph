"""Post-hoc evaluation kept outside graph construction and scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from control_graph.metrics import binary_detection_metrics


@dataclass(frozen=True)
class EvaluationConfig:
    scores_path: Path
    labels_path: Path
    output_path: Path
    bootstrap: int = 1000
    seed: int = 20260910


class DetectionEvaluator:
    """Join frozen scores to a separate label sidecar and report detection metrics."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def run(self) -> dict:
        if self.config.output_path.exists():
            raise FileExistsError(f"evaluation output already exists: {self.config.output_path}")
        scores = _load_scores(self.config.scores_path)
        labels = _load_labels(self.config.labels_path)
        score_ids = {record["event_id"] for record in scores}
        if score_ids != set(labels):
            raise ValueError("evaluation labels must exactly match frozen score event IDs")
        y_true = np.asarray([labels[record["event_id"]] for record in scores], dtype=np.int8)
        y_score = np.asarray([record["anomaly_score"] for record in scores], dtype=np.float64)
        source_ids = np.asarray([record["source_id"] for record in scores], dtype=str)
        metrics = binary_detection_metrics(
            y_true,
            y_score,
            source_ids,
            bootstrap=self.config.bootstrap,
            seed=self.config.seed,
        )
        report = {
            "schema": "control-graph/evaluation@1",
            "events": len(scores),
            "sources": len(set(source_ids)),
            "prevalence": float(y_true.mean()),
            "cluster_bootstrap": "source_id",
            "bootstrap_replicates": self.config.bootstrap,
            "labels_used_stage": "evaluation_only",
            **metrics,
        }
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report


def _load_scores(path: Path) -> list[dict]:
    expected = {
        "schema",
        "event_id",
        "source_id",
        "split",
        "relation",
        "anomaly_score",
        "dominant_edge",
        "edge_deviations",
    }
    records = _read_jsonl(path)
    for record in records:
        if set(record) != expected or record["schema"] != "control-graph/anomaly-score@1":
            raise ValueError("score sidecar has an invalid schema")
    if len({record["event_id"] for record in records}) != len(records):
        raise ValueError("score sidecar contains duplicate event IDs")
    return records


def _load_labels(path: Path) -> dict[str, int]:
    records = _read_jsonl(path)
    labels = {}
    for record in records:
        if set(record) != {"event_id", "label"} or record["label"] not in {0, 1}:
            raise ValueError("label sidecar records require event_id and binary label")
        if record["event_id"] in labels:
            raise ValueError("label sidecar contains duplicate event IDs")
        labels[record["event_id"]] = int(record["label"])
    return labels


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"JSONL must contain at least one object: {path}")
    return records
