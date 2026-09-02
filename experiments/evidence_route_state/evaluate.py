"""Post-hoc evaluation of frozen, label-free route-state scores."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

SCORE_NAMES = (
    "captured_posterior",
    "independent_token_posterior",
    "one_hop_posterior",
    "endpoint_rewire_posterior",
    "weight_shuffle_posterior",
    "functional_route_collapse",
    "attention_route_collapse",
    "route_contraction",
    "unrooted_takeover",
    "confidence",
)
PAIRED_CONTROLS = (
    "independent_token_posterior",
    "one_hop_posterior",
    "endpoint_rewire_posterior",
    "weight_shuffle_posterior",
    "functional_route_collapse",
    "attention_route_collapse",
)


def freeze_scores(
    records: Sequence[Mapping[str, object]], output: str | Path
) -> dict[str, np.ndarray]:
    """Write detector outputs before the evaluation dataset exposes labels."""

    arrays: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "source_id": [],
        "token_index": [],
        "prediction_position": [],
        "response_length": [],
        "valid": [],
        "contraction": [],
        "takeover": [],
        **{name: [] for name in SCORE_NAMES},
    }
    for record in records:
        count = len(np.asarray(record["captured_posterior"]))
        arrays["sample_id"].append(np.repeat(str(record["sample_id"]), count))
        arrays["source_id"].append(np.repeat(str(record["source_id"]), count))
        arrays["token_index"].append(np.arange(count, dtype=np.int32))
        arrays["prediction_position"].append(
            np.asarray(record["prediction_position"], dtype=np.int32)
        )
        arrays["response_length"].append(np.full(count, count, dtype=np.int32))
        arrays["valid"].append(np.asarray(record["valid"], dtype=bool))
        arrays["contraction"].append(
            np.asarray(record["contraction"], dtype=np.float32)
        )
        arrays["takeover"].append(np.asarray(record["takeover"], dtype=np.float32))
        for name in SCORE_NAMES:
            arrays[name].append(np.asarray(record[name], dtype=np.float32))

    frozen = {name: np.concatenate(values) for name, values in arrays.items()}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **frozen)
    return frozen


def load_labels(
    records: Sequence[Mapping[str, object]], frozen: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Open token labels only after score arrays have been frozen to disk."""

    by_split: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        by_split.setdefault(str(record["split_root"]), []).append(record)

    stores = {}
    for split_root, selected in by_split.items():
        dataset = open_research_dataset(
            split_root,
            device="cpu",
            retain_embedded_labels=True,
        )
        stores[split_root] = (
            dataset,
            dataset.prepare_evaluation_labels(
                [str(record["sample_id"]) for record in selected]
            ),
        )

    labels = []
    offset = 0
    for record in records:
        dataset, store = stores[str(record["split_root"])]
        sample = dataset[str(record["sample_id"])]
        attention = sample.attention()
        try:
            response_ids = attention.token_ids[int(attention.response_idx) :]
            expected = np.asarray(record["response_token_ids"], dtype=np.int64)
            actual = response_ids.detach().cpu().numpy().astype(np.int64)
            if not np.array_equal(actual, expected):
                raise ValueError("frozen score and evaluation response differ")
            labels.append(store.response_labels(sample).cpu().numpy().astype(bool))
        finally:
            sample.release_attention()
        offset += len(expected)

    label = np.concatenate(labels)
    if offset != len(frozen["sample_id"]):
        raise ValueError("frozen score and label lengths differ")
    return label


def score_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if np.unique(label).size < 2:
        return {"auroc": None, "average_precision": None, "lift": None}
    average_precision = float(average_precision_score(label, score))
    prevalence = float(label.mean())
    return {
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": average_precision,
        "lift": average_precision / prevalence,
    }


def source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, list[float] | int]:
    """Resample whole sources so long responses are not treated as independent."""

    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                (
                    roc_auc_score(label[index], score[index]),
                    average_precision_score(label[index], score[index]),
                )
            )
    if not values:
        return {
            "replicates": 0,
            "auroc_ci95": [None, None],
            "average_precision_ci95": [None, None],
        }
    interval = np.quantile(np.asarray(values), (0.025, 0.975), axis=0)
    return {
        "replicates": len(values),
        "auroc_ci95": interval[:, 0].tolist(),
        "average_precision_ci95": interval[:, 1].tolist(),
    }


def paired_source_bootstrap(
    label: np.ndarray,
    primary: np.ndarray,
    control: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap primary-minus-control metrics on exactly the same sources."""

    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size < 2:
            continue
        values.append(
            (
                roc_auc_score(label[index], primary[index])
                - roc_auc_score(label[index], control[index]),
                average_precision_score(label[index], primary[index])
                - average_precision_score(label[index], control[index]),
            )
        )
    if not values:
        return {
            "replicates": 0,
            "auroc_difference_ci95": [None, None],
            "average_precision_difference_ci95": [None, None],
        }
    interval = np.quantile(np.asarray(values), (0.025, 0.975), axis=0)
    return {
        "replicates": len(values),
        "auroc_difference_ci95": interval[:, 0].tolist(),
        "average_precision_difference_ci95": interval[:, 1].tolist(),
    }


def source_equal_mean(value: np.ndarray, source_id: np.ndarray) -> float | None:
    """Average tokens within source before averaging sources."""

    groups = np.unique(source_id)
    if not len(groups):
        return None
    return float(np.mean([value[source_id == group].mean() for group in groups]))


def legitimate_focus_audit(
    label: np.ndarray,
    frozen: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Audit correct tokens whose locked route-collapse score is already high."""

    valid = np.asarray(frozen["valid"], dtype=bool)
    valid &= np.isfinite(frozen["captured_posterior"])
    correct = valid & ~label
    narrow = correct & (frozen["functional_route_collapse"] >= 0.9)

    def mean(name: str, selected: np.ndarray) -> float | None:
        return source_equal_mean(
            frozen[name][selected],
            frozen["source_id"][selected],
        )

    return {
        "definition": "correct token and equation-locked functional collapse >= 0.9",
        "threshold_fixed_before_labels": 0.9,
        "tokens": int(narrow.sum()),
        "sources": int(np.unique(frozen["source_id"][narrow]).size),
        "captured_posterior_source_equal_mean": mean("captured_posterior", narrow),
        "captured_rate_at_0.5_source_equal_mean": source_equal_mean(
            (frozen["captured_posterior"] >= 0.5)[narrow],
            frozen["source_id"][narrow],
        ),
        "takeover_source_equal_mean": mean("takeover", narrow),
        "all_correct_captured_posterior_source_equal_mean": mean(
            "captured_posterior", correct
        ),
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
    """Join frozen scores to labels and report fixed score directions."""

    with np.load(frozen_path) as stored:
        frozen = {name: stored[name] for name in stored.files}
    label = load_labels(records, frozen)
    valid = np.asarray(frozen["valid"], dtype=bool)
    primary_valid = valid & np.isfinite(frozen["captured_posterior"])
    evaluated_label = label[primary_valid]
    detection = {}
    for index, name in enumerate(SCORE_NAMES):
        score_valid = valid & np.isfinite(frozen[name])
        score_label = label[score_valid]
        score = frozen[name][score_valid]
        result = score_metrics(score_label, score)
        result.update(
            evaluated_tokens=int(score_valid.sum()),
            evaluated_positives=int(score_label.sum()),
            prevalence=float(score_label.mean()) if len(score_label) else None,
        )
        if bootstrap and result["auroc"] is not None:
            result.update(
                source_bootstrap(
                    score_label,
                    score,
                    frozen["source_id"][score_valid],
                    bootstrap,
                    seed + index,
                )
            )
        detection[name] = result

    paired_comparisons = {}
    for index, name in enumerate(PAIRED_CONTROLS):
        common = valid & np.isfinite(frozen["captured_posterior"])
        common &= np.isfinite(frozen[name])
        common_label = label[common]
        primary = frozen["captured_posterior"][common]
        control = frozen[name][common]
        primary_metrics = score_metrics(common_label, primary)
        control_metrics = score_metrics(common_label, control)
        comparison = {
            "evaluated_tokens": int(common.sum()),
            "primary_auroc": primary_metrics["auroc"],
            "control_auroc": control_metrics["auroc"],
            "auroc_difference": None,
            "primary_average_precision": primary_metrics["average_precision"],
            "control_average_precision": control_metrics["average_precision"],
            "average_precision_difference": None,
        }
        if primary_metrics["auroc"] is not None:
            comparison["auroc_difference"] = (
                primary_metrics["auroc"] - control_metrics["auroc"]
            )
            comparison["average_precision_difference"] = (
                primary_metrics["average_precision"]
                - control_metrics["average_precision"]
            )
            if bootstrap:
                comparison.update(
                    paired_source_bootstrap(
                        common_label,
                        primary,
                        control,
                        frozen["source_id"][common],
                        bootstrap,
                        seed + 100 + index,
                    )
                )
        paired_comparisons[name] = comparison

    report: dict[str, object] = {
        "task_type": task_type,
        "samples": len(records),
        "sources": int(np.unique(frozen["source_id"]).size),
        "tokens": len(label),
        "evaluated_tokens": int(primary_valid.sum()),
        "positives": int(label.sum()),
        "evaluated_positives": int(evaluated_label.sum()),
        "prevalence": float(evaluated_label.mean()) if len(evaluated_label) else None,
        "primary_score": SCORE_NAMES[0],
        "score_direction": "higher means stronger route-capture risk; never label-flipped",
        "labels_used_during": "posthoc evaluation after score freeze",
        "provenance": dict(provenance or {}),
        "detection": detection,
        "paired_primary_minus_control": paired_comparisons,
        "legitimate_narrow_focus": legitimate_focus_audit(label, frozen),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        report_path.with_name("token_scores.npz"), **frozen, label=label
    )
    return report
