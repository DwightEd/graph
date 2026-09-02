"""Open labels only after register-graph scores have been frozen."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

SCORE_NAMES = (
    "conditional_graph_energy",
    "independent_graph_energy",
    "functional_route_collapse",
    "attention_route_collapse",
    "confidence",
)
PAIRED_CONTROLS = SCORE_NAMES[1:]


def freeze_scores(
    records: Sequence[Mapping[str, object]], output: str | Path
) -> dict[str, np.ndarray]:
    """Persist scores before any evaluation store is allowed to expose labels."""

    columns: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "source_id": [],
        "token_index": [],
        "query_position": [],
        "prediction_position": [],
        "valid": [],
        **{name: [] for name in SCORE_NAMES},
    }
    for record in records:
        count = len(np.asarray(record["conditional_graph_energy"]))
        columns["sample_id"].append(np.repeat(str(record["sample_id"]), count))
        columns["source_id"].append(np.repeat(str(record["source_id"]), count))
        columns["token_index"].append(np.arange(count, dtype=np.int32))
        for name in ("query_position", "prediction_position", "valid", *SCORE_NAMES):
            columns[name].append(np.asarray(record[name]))

    frozen = {name: np.concatenate(values) for name, values in columns.items()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **frozen)
    return frozen


def load_labels(
    records: Sequence[Mapping[str, object]], frozen: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Join each frozen prediction event to its canonical response-token label."""

    stores = {}
    for split_root in {str(record["split_root"]) for record in records}:
        selected = [
            str(record["sample_id"])
            for record in records
            if str(record["split_root"]) == split_root
        ]
        dataset = open_research_dataset(
            split_root,
            device="cpu",
            retain_embedded_labels=True,
        )
        stores[split_root] = (dataset, dataset.prepare_evaluation_labels(selected))

    labels = []
    for record in records:
        dataset, store = stores[str(record["split_root"])]
        sample = dataset[str(record["sample_id"])]
        attention = sample.attention()
        try:
            response = attention.token_ids[int(attention.response_idx) :]
            expected = np.asarray(record["response_token_ids"], dtype=np.int64)
            actual = response.detach().cpu().numpy().astype(np.int64)
            if not np.array_equal(actual, expected):
                raise ValueError("frozen graph score and evaluation response differ")
            labels.append(store.response_labels(sample).cpu().numpy().astype(bool))
        finally:
            sample.release_attention()

    label = np.concatenate(labels)
    if len(label) != len(frozen["sample_id"]):
        raise ValueError("frozen score and label lengths differ")
    return label


def metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if np.unique(label).size < 2:
        return {"auroc": None, "average_precision": None, "lift": None}
    average_precision = float(average_precision_score(label, score))
    return {
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": average_precision,
        "lift": average_precision / float(label.mean()),
    }


def source_bootstrap(
    label: np.ndarray,
    first: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
    second: np.ndarray | None = None,
) -> dict[str, object]:
    """Resample complete sources for intervals and paired score differences."""

    sources = np.unique(source_id)
    rows = {source: np.flatnonzero(source_id == source) for source in sources}
    random = np.random.default_rng(seed)
    result = []
    for _ in range(replicates):
        chosen = random.choice(sources, len(sources), replace=True)
        index = np.concatenate([rows[source] for source in chosen])
        if np.unique(label[index]).size < 2:
            continue
        value = (
            roc_auc_score(label[index], first[index]),
            average_precision_score(label[index], first[index]),
        )
        if second is not None:
            value = (
                value[0] - roc_auc_score(label[index], second[index]),
                value[1] - average_precision_score(label[index], second[index]),
            )
        result.append(value)
    if not result:
        return {"replicates": 0, "auroc_ci95": [None, None], "ap_ci95": [None, None]}
    interval = np.quantile(np.asarray(result), (0.025, 0.975), axis=0)
    return {
        "replicates": len(result),
        "auroc_ci95": interval[:, 0].tolist(),
        "ap_ci95": interval[:, 1].tolist(),
    }


def evaluate_scores(
    records: Sequence[Mapping[str, object]],
    frozen_path: str | Path,
    report_path: str | Path,
    *,
    task_type: str,
    bootstrap: int = 1000,
    seed: int = 20260902,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate fixed score directions without refitting or selecting a detector."""

    with np.load(frozen_path) as stored:
        frozen = {name: stored[name] for name in stored.files}
    label = load_labels(records, frozen)
    base_valid = np.asarray(frozen["valid"], dtype=bool)
    detection = {}
    for number, name in enumerate(SCORE_NAMES):
        valid = base_valid & np.isfinite(frozen[name])
        selected_label = label[valid]
        selected_score = frozen[name][valid]
        result = metrics(selected_label, selected_score)
        result.update(
            evaluated_tokens=int(valid.sum()),
            evaluated_positives=int(selected_label.sum()),
            prevalence=float(selected_label.mean()) if len(selected_label) else None,
        )
        if bootstrap and result["auroc"] is not None:
            result.update(
                source_bootstrap(
                    selected_label,
                    selected_score,
                    frozen["source_id"][valid],
                    replicates=bootstrap,
                    seed=seed + number,
                )
            )
        detection[name] = result

    paired = {}
    for number, name in enumerate(PAIRED_CONTROLS):
        valid = base_valid & np.isfinite(frozen[SCORE_NAMES[0]])
        valid &= np.isfinite(frozen[name])
        selected_label = label[valid]
        primary = frozen[SCORE_NAMES[0]][valid]
        control = frozen[name][valid]
        first = metrics(selected_label, primary)
        second = metrics(selected_label, control)
        comparison = {
            "evaluated_tokens": int(valid.sum()),
            "auroc_difference": None,
            "average_precision_difference": None,
        }
        if first["auroc"] is not None:
            comparison["auroc_difference"] = first["auroc"] - second["auroc"]
            comparison["average_precision_difference"] = (
                first["average_precision"] - second["average_precision"]
            )
            if bootstrap:
                comparison.update(
                    source_bootstrap(
                        selected_label,
                        primary,
                        frozen["source_id"][valid],
                        replicates=bootstrap,
                        seed=seed + 100 + number,
                        second=control,
                    )
                )
        paired[name] = comparison

    primary_valid = base_valid & np.isfinite(frozen[SCORE_NAMES[0]])
    report = {
        "task_type": task_type,
        "samples": len(records),
        "sources": int(np.unique(frozen["source_id"]).size),
        "tokens": len(label),
        "positives": int(label.sum()),
        "evaluated_tokens": int(primary_valid.sum()),
        "evaluated_positives": int(label[primary_valid].sum()),
        "prevalence": float(label[primary_valid].mean())
        if primary_valid.any()
        else None,
        "primary_score": SCORE_NAMES[0],
        "score_direction": "higher means rarer conditional graph transition",
        "labels_used_during": "posthoc evaluation after score freeze",
        "provenance": dict(provenance or {}),
        "detection": detection,
        "paired_primary_minus_control": paired,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        report_path.with_name("token_scores.npz"), **frozen, label=label
    )
    return report
