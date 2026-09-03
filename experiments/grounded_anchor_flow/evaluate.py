"""Freeze anchor-flow scores, then open RAGTruth labels for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PRIMARY = "functional_response_seeded_anchor_flow"
SECONDARY = "functional_response_seeded_path_share"
CONTROLS = (
    "attention_response_seeded_anchor_flow",
    "message_response_seeded_anchor_flow",
    "functional_direct_response_share",
    "attention_response_seeded_path_share",
    "message_response_seeded_path_share",
    "relative_response_position",
    "response_length",
    "target_surprisal",
)


def read_index(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def position_adjust(score: np.ndarray, relative: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remove the ordinary response-position trend without labels."""

    adjusted = np.full(len(score), np.nan, dtype=np.float32)
    decile = np.minimum((relative * 10).astype(np.int16), 9)
    for index in range(10):
        selected = valid & np.isfinite(score) & (decile == index)
        if not selected.any():
            continue
        value = score[selected]
        center = np.median(value)
        scale = 1.4826 * np.median(np.abs(value - center))
        adjusted[selected] = (value - center) / (scale if scale >= 1e-6 else 1.0)
    return adjusted


def freeze_scores(
    inputs: Iterable[tuple[str | Path, str | Path]], task: str
) -> tuple[dict[str, np.ndarray], list[tuple[Path, Path, dict[str, Any]]]]:
    fields: dict[str, list[np.ndarray]] = {}
    records: list[tuple[Path, Path, dict[str, Any]]] = []
    for state_value, split_value in inputs:
        state_root, split_root = Path(state_value), Path(split_value)
        for row in read_index(state_root):
            if str(row["task_type"]).casefold() != task.casefold():
                continue
            artifact = torch.load(
                state_root / row["path"], map_location="cpu", weights_only=False
            )
            count = int(row["response_tokens"])
            response_index = np.arange(count, dtype=np.int32)
            values = {
                "sample_id": np.repeat(str(row["sample_id"]), count),
                "source_id": np.repeat(str(row["source_id"]), count),
                "response_index": response_index,
                "response_length": np.full(count, count, dtype=np.int32),
                "relative_response_position": (response_index + 0.5) / count,
                "target_surprisal": -artifact["target_logprob"].numpy(),
                PRIMARY: artifact[PRIMARY].numpy(),
                f"{PRIMARY}__valid": artifact["functional_anchor_valid"].numpy(),
                SECONDARY: artifact[SECONDARY].numpy(),
                f"{SECONDARY}__valid": artifact["functional_valid"].numpy(),
                "functional_direct_response_share": artifact[
                    "functional_direct_response_share"
                ].numpy(),
                "attention_response_seeded_anchor_flow": artifact[
                    "attention_response_seeded_anchor_flow"
                ].numpy(),
                "attention_response_seeded_anchor_flow__valid": artifact[
                    "attention_anchor_valid"
                ].numpy(),
                "message_response_seeded_anchor_flow": artifact[
                    "message_response_seeded_anchor_flow"
                ].numpy(),
                "message_response_seeded_anchor_flow__valid": artifact[
                    "message_anchor_valid"
                ].numpy(),
                "attention_response_seeded_path_share": artifact[
                    "attention_response_seeded_path_share"
                ].numpy(),
                "attention_response_seeded_path_share__valid": artifact[
                    "attention_valid"
                ].numpy(),
                "message_response_seeded_path_share": artifact[
                    "message_response_seeded_path_share"
                ].numpy(),
                "message_response_seeded_path_share__valid": artifact[
                    "message_valid"
                ].numpy(),
                "functional_gather_distance": artifact[
                    "functional_gather_distance"
                ].numpy(),
                "functional_future_anchor_influence": artifact[
                    "functional_future_anchor_influence"
                ].numpy(),
                "functional_anchor_concentration": artifact[
                    "functional_anchor_concentration"
                ].numpy(),
            }
            for name, value in values.items():
                fields.setdefault(name, []).append(np.asarray(value))
            records.append((state_root, split_root, row))
    if not records:
        raise ValueError(f"no {task} flow artifacts were found")

    frozen = {name: np.concatenate(parts) for name, parts in fields.items()}
    adjusted = f"{PRIMARY}_position_adjusted"
    frozen[adjusted] = position_adjust(
        frozen[PRIMARY],
        frozen["relative_response_position"],
        frozen[f"{PRIMARY}__valid"],
    )
    frozen[f"{adjusted}__valid"] = np.isfinite(frozen[adjusted])
    return frozen, records


def load_labels(records: list[tuple[Path, Path, dict[str, Any]]]) -> np.ndarray:
    from research_dataset import open_research_dataset

    labels: list[np.ndarray] = []
    by_split: dict[Path, list[dict[str, Any]]] = {}
    for _state, split, row in records:
        by_split.setdefault(split, []).append(row)
    for split, rows in by_split.items():
        dataset = open_research_dataset(
            split, device="cpu", retain_embedded_labels=True
        )
        prepared = dataset.prepare_evaluation_labels(
            [str(row["sample_id"]) for row in rows]
        )
        for row in rows:
            sample = dataset[str(row["sample_id"])]
            value = np.asarray(prepared.response_labels(sample).cpu(), dtype=bool)
            sample.release_attention()
            if len(value) != int(row["response_tokens"]):
                raise ValueError("flow score and response label lengths differ")
            labels.append(value)
    return np.concatenate(labels)


def bootstrap_metrics(
    label: np.ndarray,
    first: np.ndarray,
    source: np.ndarray,
    replicates: int,
    seed: int,
    second: np.ndarray | None = None,
) -> tuple[list[float | None], list[float | None], int]:
    """Source-cluster intervals for one score or a paired score difference."""

    groups = np.unique(source)
    rows = {group: np.flatnonzero(source == group) for group in groups}
    random = np.random.default_rng(seed)
    result: list[tuple[float, float]] = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size != 2:
            continue
        auroc = roc_auc_score(label[index], first[index])
        ap = average_precision_score(label[index], first[index])
        if second is not None:
            auroc -= roc_auc_score(label[index], second[index])
            ap -= average_precision_score(label[index], second[index])
        result.append((float(auroc), float(ap)))
    if not result:
        return [None, None], [None, None], 0
    values = np.asarray(result)
    return (
        [float(x) for x in np.quantile(values[:, 0], (0.025, 0.975))],
        [float(x) for x in np.quantile(values[:, 1], (0.025, 0.975))],
        len(values),
    )


def score_mask(name: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
    score = np.asarray(arrays[name], dtype=np.float64)
    valid = np.isfinite(score)
    if f"{name}__valid" in arrays:
        valid &= np.asarray(arrays[f"{name}__valid"], dtype=bool)
    return valid


def metric(
    name: str,
    arrays: dict[str, np.ndarray],
    label: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    score = np.asarray(arrays[name], dtype=np.float64)
    valid = score_mask(name, arrays)
    current_label, current_score = label[valid], score[valid]
    result: dict[str, Any] = {
        "tokens": int(valid.sum()),
        "positives": int(current_label.sum()),
        "auroc": None,
        "average_precision": None,
        "auroc_ci95": [None, None],
        "average_precision_ci95": [None, None],
        "bootstrap_successful": 0,
    }
    if np.unique(current_label).size != 2:
        return result
    result["auroc"] = float(roc_auc_score(current_label, current_score))
    result["average_precision"] = float(
        average_precision_score(current_label, current_score)
    )
    if bootstrap:
        auroc_ci, ap_ci, successful = bootstrap_metrics(
            current_label,
            current_score,
            arrays["source_id"][valid],
            bootstrap,
            seed,
        )
        result.update(
            auroc_ci95=auroc_ci,
            average_precision_ci95=ap_ci,
            bootstrap_successful=successful,
        )
    return result


def paired_control(
    primary: str,
    control: str,
    arrays: dict[str, np.ndarray],
    label: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compare two capacities on the same tokens and source bootstrap draws."""

    valid = score_mask(primary, arrays) & score_mask(control, arrays)
    current_label = label[valid]
    first = np.asarray(arrays[primary], dtype=np.float64)[valid]
    second = np.asarray(arrays[control], dtype=np.float64)[valid]
    result: dict[str, Any] = {
        "control": control,
        "tokens": int(valid.sum()),
        "auroc_difference": None,
        "average_precision_difference": None,
        "auroc_difference_ci95": [None, None],
        "average_precision_difference_ci95": [None, None],
        "bootstrap_successful": 0,
    }
    if np.unique(current_label).size != 2:
        return result
    result["auroc_difference"] = float(
        roc_auc_score(current_label, first) - roc_auc_score(current_label, second)
    )
    result["average_precision_difference"] = float(
        average_precision_score(current_label, first)
        - average_precision_score(current_label, second)
    )
    if bootstrap:
        auroc_ci, ap_ci, successful = bootstrap_metrics(
            current_label,
            first,
            arrays["source_id"][valid],
            bootstrap,
            seed,
            second,
        )
        result.update(
            auroc_difference_ci95=auroc_ci,
            average_precision_difference_ci95=ap_ci,
            bootstrap_successful=successful,
        )
    return result


def evaluate(
    inputs: Iterable[tuple[str | Path, str | Path]],
    task: str,
    output: str | Path,
    *,
    bootstrap: int = 1000,
    seed: int = 20260903,
) -> dict[str, Any]:
    arrays, records = freeze_scores(inputs, task)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_name("frozen_scores.npz"), **arrays)
    label = load_labels(records)
    adjusted = f"{PRIMARY}_position_adjusted"
    names = (PRIMARY, adjusted, SECONDARY, *CONTROLS)
    report = {
        "task": task,
        "samples": len(records),
        "sources": int(np.unique(arrays["source_id"]).size),
        "tokens": len(label),
        "positives": int(label.sum()),
        "prevalence": float(label.mean()),
        "primary": PRIMARY,
        "secondary": SECONDARY,
        "score_meaning": (
            "response-seeded share of target-conditioned flow through response "
            "transit anchors; this is observed path attribution, not causal necessity"
        ),
        "metrics": {
            name: metric(name, arrays, label, bootstrap, seed + index)
            for index, name in enumerate(names)
        },
        "paired_capacity_controls": {
            control: paired_control(
                PRIMARY,
                control,
                arrays,
                label,
                bootstrap,
                seed + 100 + index,
            )
            for index, control in enumerate(
                (
                    "attention_response_seeded_anchor_flow",
                    "message_response_seeded_anchor_flow",
                )
            )
        },
        "labels_used_during": "evaluation_only_after_frozen_scores",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output.with_name("token_results.npz"), **arrays, label=label)
    return report
