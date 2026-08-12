"""Full-dataset metrics for label-blind unsupervised token scores."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


STRATUM_FIELDS = ("task_type", "data_source", "generator_model")
REQUIRED_FIELDS = (
    "sample_id",
    "source_id",
    *STRATUM_FIELDS,
    "token_index",
    "score",
    "label",
)


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics and original evaluated token records."""

    metrics: dict
    token_records: tuple[Mapping, ...]

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(self.metrics, indent=2, sort_keys=True), encoding="utf-8"
        )

        fieldnames = sorted(
            {field for record in self.token_records for field in record if field != "embedding"}
        )
        with (output / "token_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in self.token_records:
                writer.writerow({field: record[field] for field in fieldnames if field in record})


def evaluate_records(
    records: Sequence[Mapping], *, bootstraps: int = 1_000, seed: int = 0
) -> EvaluationReport:
    """Evaluate every labeled token record and sample-level top-20%-mean scores."""
    records = tuple(records)
    _validate_records(records, bootstraps)
    answer_records = _answer_records(records)
    rng = np.random.default_rng(seed)
    return EvaluationReport(
        metrics={
            "token": _level_metrics(records, bootstraps, rng),
            "answer": _level_metrics(answer_records, bootstraps, rng),
        },
        token_records=records,
    )


def compare_variants(
    variant_records: Mapping[str, Sequence[Mapping]],
    *,
    reference: str = "full",
    bootstraps: int = 1_000,
    seed: int = 0,
) -> dict:
    """Paired source-bootstrap deltas on one shared set of OOF tokens."""
    reference_records = tuple(variant_records[reference])
    reference_keys = [_record_key(record) for record in reference_records]
    reference_labels, reference_scores = _labels_and_scores(reference_records)
    reference_auroc, reference_auprc = _ranking_metrics(reference_labels, reference_scores)
    source_ids = np.asarray([record["source_id"] for record in reference_records], dtype=object)
    unique_sources = np.unique(source_ids)
    rng = np.random.default_rng(seed)
    output = {}
    for name, records in variant_records.items():
        if name == reference:
            continue
        records_by_key = {_record_key(record): record for record in records}
        if set(records_by_key) != set(reference_keys):
            raise ValueError("variants must score the same OOF tokens")
        variant_scores = np.asarray(
            [records_by_key[key]["score"] for key in reference_keys], dtype=float
        )
        variant_auroc, variant_auprc = _ranking_metrics(reference_labels, variant_scores)
        auroc_draws, auprc_draws = [], []
        for _ in range(bootstraps):
            sampled_sources = rng.choice(unique_sources, len(unique_sources), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(source_ids == source) for source in sampled_sources]
            )
            full_metrics = _ranking_metrics(reference_labels[indices], reference_scores[indices])
            variant_metrics = _ranking_metrics(reference_labels[indices], variant_scores[indices])
            if full_metrics[0] is not None:
                auroc_draws.append(full_metrics[0] - variant_metrics[0])
                auprc_draws.append(full_metrics[1] - variant_metrics[1])
        output[name] = {
            "delta_auroc": reference_auroc - variant_auroc,
            "delta_auroc_ci": list(_percentile_interval(auroc_draws)),
            "delta_auprc": reference_auprc - variant_auprc,
            "delta_auprc_ci": list(_percentile_interval(auprc_draws)),
        }
    return output


def _record_key(record: Mapping) -> tuple:
    return record["sample_id"], int(record["token_index"])


def _validate_records(records: Sequence[Mapping], bootstraps: int) -> None:
    if not records:
        raise ValueError("records must not be empty")
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"record {index} is missing required fields: {missing}")
        if record["label"] not in (0, 1):
            raise ValueError(f"record {index} label must be 0 or 1")


def _answer_records(records: Sequence[Mapping]) -> tuple[dict, ...]:
    by_sample: dict[object, list[Mapping]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)

    answers = []
    for sample_id in sorted(by_sample, key=str):
        token_records = by_sample[sample_id]
        first = token_records[0]
        for field in ("source_id", *STRATUM_FIELDS):
            if any(record[field] != first[field] for record in token_records[1:]):
                raise ValueError(f"sample {sample_id!r} has inconsistent {field}")
        scores = np.sort(np.asarray([record["score"] for record in token_records], dtype=float))
        top_count = max(1, int(np.ceil(len(scores) * 0.2)))
        answers.append(
            {
                "sample_id": sample_id,
                "source_id": first["source_id"],
                "task_type": first["task_type"],
                "data_source": first["data_source"],
                "generator_model": first["generator_model"],
                "score": float(scores[-top_count:].mean()),
                "label": int(any(record["label"] == 1 for record in token_records)),
            }
        )
    return tuple(answers)


def _level_metrics(records: Sequence[Mapping], bootstraps: int, rng: np.random.Generator) -> dict:
    metrics = {"overall": _metric_summary(records, bootstraps, rng)}
    for field in STRATUM_FIELDS:
        groups: dict[object, list[Mapping]] = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        metrics[f"by_{field}"] = {
            str(value): _metric_summary(group, bootstraps, rng)
            for value, group in sorted(groups.items(), key=lambda item: str(item[0]))
        }
    return metrics


def _metric_summary(
    records: Sequence[Mapping], bootstraps: int, rng: np.random.Generator
) -> dict:
    labels, scores = _labels_and_scores(records)
    auroc, auprc = _ranking_metrics(labels, scores)
    auroc_interval, auprc_interval = _bootstrap_intervals(records, bootstraps, rng)
    contrast, contrast_interval = _score_contrast(records, bootstraps, rng)
    negative = scores[labels == 0]
    positive = scores[labels == 1]
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "auroc_ci_low": auroc_interval[0],
        "auroc_ci_high": auroc_interval[1],
        "auprc_ci_low": auprc_interval[0],
        "auprc_ci_high": auprc_interval[1],
        "correct_score_median": None if not len(negative) else float(np.median(negative)),
        "hallucination_score_median": None if not len(positive) else float(np.median(positive)),
        "mean_score_difference": contrast,
        "mean_score_difference_ci_low": contrast_interval[0],
        "mean_score_difference_ci_high": contrast_interval[1],
    }


def _labels_and_scores(records: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([record["label"] for record in records], dtype=int),
        np.asarray([record["score"] for record in records], dtype=float),
    )


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if labels.min() == labels.max():
        return None, None
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _bootstrap_intervals(
    records: Sequence[Mapping], bootstraps: int, rng: np.random.Generator
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
    clusters: dict[object, list[Mapping]] = defaultdict(list)
    for record in records:
        clusters[record["source_id"]].append(record)
    source_ids = tuple(clusters)
    aurocs, auprcs = [], []
    for _ in range(bootstraps):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        draw = [record for source_id in sampled for record in clusters[source_id]]
        auroc, auprc = _ranking_metrics(*_labels_and_scores(draw))
        if auroc is not None:
            aurocs.append(auroc)
            auprcs.append(auprc)
    return _percentile_interval(aurocs), _percentile_interval(auprcs)


def _score_contrast(
    records: Sequence[Mapping], bootstraps: int, rng: np.random.Generator
) -> tuple[float | None, tuple[float | None, float | None]]:
    labels, scores = _labels_and_scores(records)
    if labels.min() == labels.max():
        return None, (None, None)
    observed = float(scores[labels == 1].mean() - scores[labels == 0].mean())
    clusters: dict[object, list[Mapping]] = defaultdict(list)
    for record in records:
        clusters[record["source_id"]].append(record)
    source_ids = tuple(clusters)
    draws = []
    for _ in range(bootstraps):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        draw_labels, draw_scores = _labels_and_scores(
            [record for source_id in sampled for record in clusters[source_id]]
        )
        if draw_labels.min() != draw_labels.max():
            draws.append(
                float(draw_scores[draw_labels == 1].mean() - draw_scores[draw_labels == 0].mean())
            )
    return observed, _percentile_interval(draws)


def _percentile_interval(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
