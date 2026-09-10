"""Post-hoc token-label evaluation of frozen attention-audit scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from control_graph.audit import AUDIT_SCORE_SCHEMA, load_audit_manifest
from control_graph.metrics import binary_detection_metrics

SCORE_NAMES = (
    "constraint_displacement",
    "attention_displacement",
    "negative_margin",
    "relative_position",
)
SCORE_FIELDS = {
    "schema",
    "event_id",
    "sample_id",
    "source_id",
    "split",
    "task_type",
    "response_index",
    "token_id",
    "mechanism_edges",
    "scores",
}


@dataclass(frozen=True)
class AuditEvaluationConfig:
    audit_root: Path
    scores_path: Path
    output_path: Path
    bootstrap: int = 1000
    seed: int = 20260910

    def __post_init__(self) -> None:
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be non-negative")


class AttentionAuditEvaluator:
    """Join labels after scoring and compare onset and continuation cohorts."""

    def __init__(self, config: AuditEvaluationConfig, *, progress: bool = True) -> None:
        self.config = config
        self.progress = progress

    def run(self) -> dict:
        if self.config.output_path.exists():
            raise FileExistsError(
                f"audit evaluation output already exists: {self.config.output_path}"
            )
        if not self.config.scores_path.is_file():
            raise FileNotFoundError(f"frozen audit scores do not exist: {self.config.scores_path}")
        root = self.config.audit_root.resolve()
        entries = load_audit_manifest(root)
        entry_by_identity = {
            (entry["split"], entry["task_type"], entry["sample_id"]): entry
            for entry in entries
        }
        if len(entry_by_identity) != len(entries):
            raise ValueError("attention-audit manifest contains duplicate sample identities")

        labels, previous, sources = [], [], []
        score_columns = {name: [] for name in SCORE_NAMES}
        seen_events: set[str] = set()
        label_cache: dict[tuple[str, str, str], np.ndarray] = {}
        total_bytes = self.config.scores_path.stat().st_size
        progress = tqdm(
            total=total_bytes,
            desc="join frozen scores and labels",
            unit="B",
            unit_scale=True,
            disable=not self.progress,
        )
        with self.config.scores_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                progress.update(len(line.encode("utf-8")))
                if not line.strip():
                    continue
                record = _validated_score(json.loads(line), self.config.scores_path, line_number)
                if record["event_id"] in seen_events:
                    raise ValueError(f"duplicate frozen score event_id: {record['event_id']}")
                seen_events.add(record["event_id"])
                identity = (record["split"], record["task_type"], record["sample_id"])
                entry = entry_by_identity.get(identity)
                if entry is None or entry["source_id"] != record["source_id"]:
                    raise ValueError(f"score event is not aligned to index.json: {record['event_id']}")
                sample_labels = label_cache.get(identity)
                if sample_labels is None:
                    sample_labels = _load_labels(root, entry)
                    label_cache[identity] = sample_labels
                response_index = record["response_index"]
                if not 0 <= response_index < len(sample_labels):
                    raise ValueError(f"score response index is outside labels: {record['event_id']}")
                labels.append(int(sample_labels[response_index]))
                previous.append(int(sample_labels[response_index - 1]) if response_index else -1)
                sources.append(record["source_id"])
                for name in SCORE_NAMES:
                    score_columns[name].append(float(record["scores"][name]))
        progress.close()
        if not labels:
            raise ValueError("frozen audit score JSONL contains no events")

        label_array = np.asarray(labels, dtype=np.int8)
        previous_array = np.asarray(previous, dtype=np.int8)
        source_array = np.asarray(sources, dtype=str)
        score_arrays = {
            name: np.asarray(values, dtype=np.float64) for name, values in score_columns.items()
        }
        cohorts = {
            "all": np.isin(label_array, (0, 1)),
            "onset": (label_array == 0) | ((label_array == 1) & (previous_array == 0)),
            "continuation": (label_array == 0) | (
                (label_array == 1) & (previous_array == 1)
            ),
        }
        report = {
            "schema": "control-graph/audit-evaluation@1",
            "tokens_joined": len(labels),
            "labeled_tokens": int(np.isin(label_array, (0, 1)).sum()),
            "sources": len(set(sources)),
            "cluster_bootstrap": "source_id",
            "point_estimate_weighting": "equal_source",
            "bootstrap_replicates": self.config.bootstrap,
            "labels_used_stage": "evaluation_only",
            "cohorts": {
                name: _evaluate_cohort(
                    label_array,
                    source_array,
                    score_arrays,
                    mask,
                    bootstrap=self.config.bootstrap,
                    seed=self.config.seed,
                )
                for name, mask in cohorts.items()
            },
        }
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report


def _validated_score(record: dict, path: Path, line_number: int) -> dict:
    if (
        not isinstance(record, dict)
        or set(record) != SCORE_FIELDS
        or record.get("schema") != AUDIT_SCORE_SCHEMA
        or not isinstance(record.get("scores"), dict)
        or set(record["scores"]) != set(SCORE_NAMES)
        or type(record.get("response_index")) is not int
        or record["response_index"] < 0
    ):
        raise ValueError(f"invalid frozen audit score at {path}:{line_number}")
    if any(
        type(record["scores"][name]) not in {int, float}
        or not np.isfinite(record["scores"][name])
        for name in SCORE_NAMES
    ):
        raise ValueError(f"non-finite frozen audit score at {path}:{line_number}")
    return record


def _load_labels(root: Path, entry: dict) -> np.ndarray:
    trace_path = (root / entry["path"]).resolve()
    if not trace_path.is_relative_to(root):
        raise ValueError(f"attention trace escapes the audit root: {entry['path']}")
    path = trace_path.with_suffix(".labels.npz")
    if not path.is_file():
        raise FileNotFoundError(f"attention-audit label sidecar does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive.files:
            raise ValueError(f"attention-audit label sidecar has no labels array: {path}")
        labels = np.asarray(archive["labels"])
    if labels.shape != (entry["response_tokens"],) or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError(f"labels must cover the full response and use -1/0/1: {path}")
    return labels.astype(np.int8, copy=False)


def _evaluate_cohort(
    labels: np.ndarray,
    source_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    selected_labels = labels[mask]
    positives = int((selected_labels == 1).sum())
    negatives = int((selected_labels == 0).sum())
    result = {
        "tokens": int(mask.sum()),
        "positives": positives,
        "negatives": negatives,
        "token_prevalence": positives / len(selected_labels) if len(selected_labels) else None,
        "source_balanced_prevalence": _balanced_prevalence(
            selected_labels, source_ids[mask]
        ),
        "scores": {},
    }
    if not positives or not negatives:
        result["scores"] = {
            name: {
                "auroc": None,
                "auprc": None,
                "valid_bootstrap_replicates": 0,
                "confidence_intervals": {"auroc": None, "auprc": None},
            }
            for name in SCORE_NAMES
        }
        return result
    for name, values in scores.items():
        result["scores"][name] = binary_detection_metrics(
            selected_labels,
            values[mask],
            source_ids[mask],
            bootstrap=bootstrap,
            seed=seed,
            source_balanced=True,
        )
    return result


def _balanced_prevalence(labels: np.ndarray, source_ids: np.ndarray) -> float | None:
    if not len(labels):
        return None
    _, inverse, counts = np.unique(source_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return float(np.average(labels, weights=weights))
