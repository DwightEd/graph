"""Evaluation-only label join for frozen unsupervised token scores."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .score import SCORE_COMPONENTS, load_score_records


def _ranking(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {
            "n": int(len(labels)), "positives": int(labels.sum()),
            "prevalence": float(labels.mean()) if len(labels) else None,
            "auroc": None, "auprc": None,
        }
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "correct_median": float(np.median(scores[labels == 0])),
        "hallucination_median": float(np.median(scores[labels == 1])),
    }


def _attach_labels(dataset, records):
    store = dataset.labels()
    cache = {}
    output = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in cache:
            sample = dataset[sample_id]
            cache[sample_id] = store.response_labels(sample).cpu().numpy()
            sample.release_attention()
        index = int(record["token_index"])
        if not 0 <= index < len(cache[sample_id]):
            raise ValueError("score token index is outside the response")
        output.append({**record, "label": int(cache[sample_id][index])})
    expected = {
        (sample_id, index)
        for sample_id in dataset.sample_ids
        for index in range(dataset[sample_id].attention().num_response_tokens)
    }
    observed = {(str(row["sample_id"]), int(row["token_index"])) for row in output}
    if observed != expected:
        raise ValueError("frozen score artifact does not cover the canonical split exactly")
    return output


def _response_records(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    output = []
    for sample_id, rows in grouped.items():
        scores = np.sort(np.asarray([row["score"] for row in rows], dtype=float))
        top_count = max(1, int(np.ceil(len(scores) * 0.2)))
        output.append({
            "sample_id": sample_id,
            "label": int(any(row["label"] for row in rows)),
            "score": float(scores[-top_count:].mean()),
        })
    return output


def evaluate_scores(dataset, *, score_path, output_path):
    """Load labels only after embeddings and anomaly scores are frozen."""
    records = _attach_labels(dataset, load_score_records(score_path))
    labels = np.asarray([row["label"] for row in records])
    report = {
        "schema": "attention-graph-evaluation-v1",
        "labels_read_during": "evaluation_only",
        "token": {"overall": _ranking(labels, [row["score"] for row in records])},
        "components": {},
    }
    for name in SCORE_COMPONENTS:
        field = f"residual_{name}"
        if field in records[0]:
            report["components"][name] = _ranking(labels, [row[field] for row in records])
    for field in ("task_type", "data_source", "generator_model"):
        groups = defaultdict(list)
        for row in records:
            groups[str(row.get(field))].append(row)
        report["token"][f"by_{field}"] = {
            key: _ranking([row["label"] for row in group], [row["score"] for row in group])
            for key, group in sorted(groups.items())
        }
    response = _response_records(records)
    report["response"] = _ranking(
        [row["label"] for row in response], [row["score"] for row in response]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
