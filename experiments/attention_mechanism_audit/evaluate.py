"""Freeze label-free OOF scores, then evaluate and audit them with labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .capture import EVIDENCE, HISTORY, ROLE_NAMES, SELF
from .collect import SCHEMA, VERSION, load_index
from .data import canonical_task_type
from .detect import SCORE_DEFINITIONS, SCORE_NAMES, factorial_contrasts, score_records
from .visualize import plot_population, plot_sample_dashboard

SCORE_ORDER = SCORE_NAMES
PRIMARY_SCORE = "mechanism_innovation"
CONTROL_SCORES = ("static_state", "confidence")
ONSET_AUDIT_NAMES = {
    "edge_route_balance_mean",
    "edge_route_velocity_mean",
    "source_dispersion_mean",
    "edge_head_role_jsd_mean",
    "source_coherence_evidence_mean",
    "source_coherence_response_history_mean",
    "head_coherence_evidence_mean",
    "head_coherence_response_history_mean",
    "causal_evidence_support",
    "causal_history_support",
    "causal_interaction",
    "remaining_context_margin",
}
_EPS = 1e-12


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(path: str | Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _layer_reductions(value: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Reduce a label-free ``[layer, token]`` measurement."""

    width = max(value.shape[0] // 3, 1)
    early = value[:width].mean(axis=0)
    late = value[-width:].mean(axis=0)
    return {
        f"{name}_mean": value.mean(axis=0).astype(np.float32),
        f"{name}_early": early.astype(np.float32),
        f"{name}_late": late.astype(np.float32),
        f"{name}_layer_shift": (late - early).astype(np.float32),
    }


def _aggregate_role_share(value: np.ndarray) -> np.ndarray:
    """Aggregate heads by their actual mass, then normalize over roles."""

    role = value.sum(axis=2)
    return role / np.maximum(role.sum(axis=-1, keepdims=True), _EPS)


def _head_role_jsd(value: np.ndarray) -> np.ndarray:
    """Head disagreement over non-self routes, normalized to ``[0,1]``."""

    nonself = value[..., :SELF]
    mass = nonself.sum(axis=-1)
    valid = mass > _EPS
    probability = nonself / np.maximum(mass[..., None], _EPS)
    count = np.maximum(valid.sum(axis=2), 1)
    mean = (probability * valid[..., None]).sum(axis=2) / count[..., None]
    head_entropy = -(
        probability * np.log(np.maximum(probability, _EPS))
    ).sum(axis=-1)
    mean_entropy = -(mean * np.log(np.maximum(mean, _EPS))).sum(axis=-1)
    jsd = mean_entropy - (head_entropy * valid).sum(axis=2) / count
    return np.maximum(jsd, 0) / np.log(SELF)


def _route_velocity(route_share: np.ndarray) -> np.ndarray:
    """Total-variation change of the non-self route composition."""

    velocity = np.zeros(route_share.shape[:2], dtype=np.float64)
    velocity[:, 1:] = 0.5 * np.abs(np.diff(route_share, axis=1)).sum(axis=-1)
    return velocity


def layer_audit_metrics(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return head-resolved measurements used by plots and post-hoc audit."""

    trace = artifact["trace"]
    measurements = {
        "attention": _array(trace["role_attention_mass"]).astype(np.float64),
        "edge": _array(trace["edge_role_energy"]).astype(np.float64),
        "write": _array(trace["head_role_write_norm"]).astype(np.float64),
    }
    result: dict[str, np.ndarray] = {}
    family_shares = {}
    for family, value in measurements.items():
        share = _aggregate_role_share(value)
        family_shares[family] = share
        for role_index, role in enumerate(ROLE_NAMES):
            result[f"{family}_{role}_share"] = share[..., role_index]
        nonself = value[..., :SELF].sum(axis=2)
        nonself /= np.maximum(nonself.sum(axis=-1, keepdims=True), _EPS)
        result[f"{family}_route_balance"] = (
            nonself[..., HISTORY] - nonself[..., EVIDENCE]
        )
        result[f"{family}_head_role_jsd"] = _head_role_jsd(value)
        result[f"{family}_route_velocity"] = _route_velocity(nonself)
    edge_role = measurements["edge"].sum(axis=2)
    write_role = measurements["write"].sum(axis=2)
    within_head_coherence = write_role / np.maximum(edge_role, _EPS)
    for role_index, role in enumerate(ROLE_NAMES):
        result[f"source_coherence_{role}"] = within_head_coherence[
            ..., role_index
        ]
        result[f"edge_attention_gain_{role}"] = (
            family_shares["edge"][..., role_index]
            - family_shares["attention"][..., role_index]
        )
        result[f"write_edge_gain_{role}"] = (
            family_shares["write"][..., role_index]
            - family_shares["edge"][..., role_index]
        )
    entropy = _array(trace["head_source_entropy"]).astype(np.float64)
    result["source_dispersion"] = entropy.mean(axis=2)
    result["source_dispersion_head_std"] = entropy.std(axis=2)
    coherence = _array(trace["role_head_coherence"]).astype(np.float64)
    for role_index, role in enumerate(ROLE_NAMES):
        result[f"head_coherence_{role}"] = coherence[..., role_index]
    return result


def token_audit_metrics(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build rich label-free audit arrays; labels never select these quantities."""

    metrics: dict[str, np.ndarray] = {}
    for name, value in layer_audit_metrics(artifact).items():
        metrics.update(_layer_reductions(value, name))
    contrasts = factorial_contrasts(artifact).astype(np.float32)
    metrics.update(
        {
            "causal_evidence_support": contrasts[:, 0],
            "causal_history_support": contrasts[:, 1],
            "causal_interaction": contrasts[:, 2],
            "remaining_context_margin": _array(
                artifact["score_inputs"]["no_evidence_history_margin"]
            ).astype(np.float32),
        }
    )
    return metrics


def _binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prevalence = float(label.mean())
    auprc = float(average_precision_score(label, score))
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": auprc,
        "auprc_lift": auprc / prevalence,
    }


def _source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
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
    values = np.asarray(values)
    if not len(values):
        return {
            "replicates": 0,
            "auroc_low": None,
            "auroc_high": None,
            "auprc_low": None,
            "auprc_high": None,
        }
    return {
        "replicates": int(len(values)),
        "auroc_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_high": float(np.quantile(values[:, 1], 0.975)),
    }


def detection_summary(
    label: np.ndarray,
    scores: Mapping[str, np.ndarray],
    source_id: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate frozen directions; labels never flip or refit a score."""

    if np.unique(label).size != 2:
        return {
            name: {
                "auroc": None,
                "auprc": None,
                "auprc_lift": None,
                "auroc_ci95": [None, None],
                "auprc_ci95": [None, None],
            }
            for name in SCORE_ORDER
        }
    results = {}
    for offset, name in enumerate(SCORE_ORDER):
        result: dict[str, Any] = _binary_metrics(label, scores[name])
        if bootstrap:
            interval = _source_bootstrap(
                label,
                scores[name],
                source_id,
                replicates=bootstrap,
                seed=seed + offset,
            )
            result.update(
                {
                    "auroc_ci95": [interval["auroc_low"], interval["auroc_high"]],
                    "auprc_ci95": [interval["auprc_low"], interval["auprc_high"]],
                    "bootstrap_replicates": interval["replicates"],
                }
            )
        else:
            result.update({"auroc_ci95": [None, None], "auprc_ci95": [None, None]})
        results[name] = result
    return results


def _position_match_design(
    label: np.ndarray,
    sample_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    relative = np.minimum(
        ((token_index + 0.5) * 10 / response_length).astype(np.int16), 9
    )
    absolute = token_index // position_bin
    cells: dict[tuple[str, int, int], list[int]] = {}
    for index, key in enumerate(zip(sample_id, absolute, relative)):
        cells.setdefault(key, []).append(index)
    matched = []
    for (sample, _absolute, _relative), rows in cells.items():
        rows = np.asarray(rows)
        positive = rows[label[rows]]
        negative = rows[~label[rows]]
        if len(positive) and len(negative):
            weight = len(positive) * len(negative) / len(rows)
            matched.append((str(sample), positive, negative, float(weight)))
    return matched


def _source_interval(
    source_effects: Mapping[str, Sequence[float]], *, bootstrap: int, seed: int
) -> tuple[float | None, list[float | None], int]:
    values = np.asarray(
        [np.mean(current) for current in source_effects.values()], dtype=np.float64
    )
    if not len(values):
        return None, [None, None], 0
    interval: list[float | None] = [None, None]
    if bootstrap:
        random = np.random.default_rng(seed)
        draws = random.choice(values, (bootstrap, len(values)), replace=True).mean(1)
        interval = [float(x) for x in np.quantile(draws, (0.025, 0.975))]
    return float(values.mean()), interval, int(len(values))


def _position_matched_difference(
    value: np.ndarray,
    matched: Sequence[tuple[str, np.ndarray, np.ndarray, float]],
    sample_source: Mapping[str, str],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    by_sample: dict[str, list[tuple[float, float]]] = {}
    for sample, positive, negative, weight in matched:
        effect = float(value[positive].mean() - value[negative].mean())
        by_sample.setdefault(sample, []).append((effect, weight))
    by_source: dict[str, list[float]] = {}
    for sample, rows in by_sample.items():
        effects = np.asarray([effect for effect, _weight in rows])
        weights = np.asarray([weight for _effect, weight in rows])
        by_source.setdefault(sample_source[sample], []).append(
            float(np.average(effects, weights=weights))
        )
    effect, interval, sources = _source_interval(
        by_source, bootstrap=bootstrap, seed=seed
    )
    return {
        "hallucinated_minus_correct": effect,
        "ci95": interval,
        "sources": sources,
        "matched_samples": int(len(by_sample)),
        "matched_cells": int(len(matched)),
    }


def _onset_difference_in_difference(
    value: np.ndarray,
    label: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
    window: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Compare route change at hallucination onset with local correct pivots."""

    by_source: dict[str, list[float]] = {}
    events = controls = 0
    for sample in np.unique(sample_id):
        rows = np.flatnonzero(sample_id == sample)
        order = rows[np.argsort(token_index[rows])]
        y = label[order]
        if len(order) < 2 * window + 1:
            continue
        relative = np.minimum(
            ((token_index[order] + 0.5) * 10 / response_length[order]).astype(int),
            9,
        )
        absolute = token_index[order] // position_bin
        onset = np.flatnonzero(y & np.r_[True, ~y[:-1]])
        correct = np.flatnonzero(~y)
        for pivot in onset:
            if pivot < window or pivot + window > len(order):
                continue
            before = np.arange(pivot - window, pivot)
            after = np.arange(pivot, pivot + window)
            if not (~y[before]).all() or not y[after].all():
                continue
            candidates = [
                current
                for current in correct
                if current >= window
                and current + window <= len(order)
                and (~y[current - window : current + window]).all()
                and relative[current] == relative[pivot]
                and absolute[current] == absolute[pivot]
            ]
            if not candidates:
                continue
            event_change = value[order[after]].mean() - value[order[before]].mean()
            control_changes = [
                value[order[current : current + window]].mean()
                - value[order[current - window : current]].mean()
                for current in candidates
            ]
            source = str(source_id[order[pivot]])
            by_source.setdefault(source, []).append(
                float(event_change - np.mean(control_changes))
            )
            events += 1
            controls += len(candidates)
    effect, interval, sources = _source_interval(
        by_source, bootstrap=bootstrap, seed=seed
    )
    return {
        "onset_change_minus_matched_correct_change": effect,
        "ci95": interval,
        "sources": sources,
        "onsets": int(events),
        "matched_correct_pivots": int(controls),
        "window": int(window),
    }


def group_difference_audit(
    arrays: Mapping[str, np.ndarray],
    audit_names: Sequence[str],
    *,
    position_bin: int,
    bootstrap: int,
    seed: int,
    onset_window: int = 2,
) -> dict[str, object]:
    """Apply labels only to matched contrasts after measurements are frozen."""

    label = arrays["label"]
    matched = _position_match_design(
        label,
        arrays["sample_id"],
        arrays["token_index"],
        arrays["response_length"],
        position_bin=position_bin,
    )
    sample_source = {
        str(sample): str(source)
        for sample, source in zip(arrays["sample_id"], arrays["source_id"])
    }
    metrics = {}
    onset = {}
    for offset, name in enumerate(audit_names):
        value = arrays[name]
        metrics[name] = {
            "correct_mean": float(value[~label].mean()),
            "hallucinated_mean": float(value[label].mean()),
            **_position_matched_difference(
                value,
                matched,
                sample_source,
                bootstrap=bootstrap,
                seed=seed + offset,
            ),
        }
        if name in ONSET_AUDIT_NAMES:
            onset[name] = _onset_difference_in_difference(
                value,
                label,
                arrays["sample_id"],
                arrays["source_id"],
                arrays["token_index"],
                arrays["response_length"],
                position_bin=position_bin,
                window=onset_window,
                bootstrap=bootstrap,
                seed=seed + len(audit_names) + offset,
            )
    covered = sum(len(positive) for _sample, positive, _negative, _weight in matched)
    return {
        "role": "posthoc_mechanism_audit_not_score_selection",
        "matching": "sample_id + absolute_position_bin + relative_position_decile",
        "aggregation": "matched cells -> response -> equal source",
        "bootstrap_unit": "source_id",
        "position_bin": int(position_bin),
        "covered_hallucinated_tokens": int(covered),
        "hallucinated_token_coverage": float(covered / max(label.sum(), 1)),
        "metrics": metrics,
        "onset_difference_in_difference": onset,
    }


def _load_manifest(trace_root: Path, task_type: str | None = None) -> dict:
    manifest = json.loads((trace_root / "manifest.json").read_text(encoding="utf-8"))
    valid = (
        manifest.get("schema") == SCHEMA
        and manifest.get("version") == VERSION
        and (task_type is None or task_type in manifest.get("task_types", []))
    )
    if not valid:
        raise ValueError(f"mechanism-state manifest does not match v{VERSION}")
    return manifest


def _pool_records(
    inputs: Iterable[tuple[str | Path, str | Path]], task_type: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool both physical shards as metadata before loading artifacts or labels."""

    records: list[dict[str, Any]] = []
    manifests = []
    for shard, (trace_value, split_value) in enumerate(inputs):
        trace_root, split_root = Path(trace_value), Path(split_value)
        manifests.append(_load_manifest(trace_root, task_type))
        current = []
        for row in load_index(trace_root):
            if canonical_task_type(row["task_type"]) != task_type:
                continue
            current.append(
                {
                    **row,
                    "sample_id": str(row["sample_id"]),
                    "source_id": str(row["source_id"]),
                    "path": trace_root / "samples" / row["path"],
                    "split_root": split_root,
                    "physical_shard": shard,
                }
            )
        if not current:
            raise ValueError(f"no {task_type} samples in {trace_root}")
        records.extend(current)
    return records, manifests


def _concatenate_label_free(
    records: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Join OOF scores and audit arrays in immutable record order."""

    output: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "source_id": [],
        "physical_shard": [],
        "token_index": [],
        "response_length": [],
        **{name: [] for name in SCORE_ORDER},
    }
    audit_names: tuple[str, ...] | None = None
    for record in records:
        sample = str(record["sample_id"])
        if sample not in scores:
            raise ValueError(f"detector did not score sample {sample}")
        current_scores = scores[sample]
        count = len(current_scores[PRIMARY_SCORE])
        if any(len(current_scores[name]) != count for name in SCORE_ORDER):
            raise ValueError(f"detector score length mismatch for sample {sample}")
        expected = record.get("response_tokens")
        if expected is not None and int(expected) != count:
            raise ValueError(f"artifact/index length mismatch for sample {sample}")
        audit = token_audit_metrics(_load_artifact(record["path"]))
        if any(len(value) != count for value in audit.values()):
            raise ValueError(f"audit length mismatch for sample {sample}")
        if audit_names is None:
            audit_names = tuple(audit)
            output.update({name: [] for name in audit_names})
        elif tuple(audit) != audit_names:
            raise ValueError("audit measurement schema changed between samples")
        output["sample_id"].append(np.repeat(sample, count))
        output["source_id"].append(np.repeat(str(record["source_id"]), count))
        output["physical_shard"].append(
            np.full(count, int(record["physical_shard"]), dtype=np.int8)
        )
        output["token_index"].append(np.arange(count, dtype=np.int32))
        output["response_length"].append(np.full(count, count, dtype=np.int32))
        for name in SCORE_ORDER:
            output[name].append(np.asarray(current_scores[name], dtype=np.float32))
        for name, value in audit.items():
            output[name].append(np.asarray(value, dtype=np.float32))
    if set(scores) != {str(record["sample_id"]) for record in records}:
        raise ValueError("detector returned scores for an unknown sample")
    return {name: np.concatenate(values) for name, values in output.items()}, (
        audit_names or ()
    )


def _write_frozen(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_labels(
    records: Sequence[Mapping[str, Any]], frozen: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Open labels only after OOF arrays have been serialized and hashed."""

    datasets: dict[int, tuple[Any, Any]] = {}
    labels = []
    offset = 0
    for record in records:
        shard = int(record["physical_shard"])
        if shard not in datasets:
            dataset = open_research_dataset(
                record["split_root"], device="cpu", retain_embedded_labels=True
            )
            ids = [
                current["sample_id"]
                for current in records
                if int(current["physical_shard"]) == shard
            ]
            datasets[shard] = (dataset, dataset.prepare_evaluation_labels(ids))
        dataset, prepared = datasets[shard]
        sample = dataset[record["sample_id"]]
        token_label = prepared.response_labels(sample).cpu().numpy().astype(bool)
        sample.release_attention()
        count = int(frozen["response_length"][offset])
        if len(token_label) != count:
            raise ValueError(
                f"frozen-score/label length mismatch for sample {record['sample_id']}"
            )
        labels.append(token_label)
        offset += count
    return np.concatenate(labels)


def build_report(
    *,
    task_type: str,
    arrays: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    detector: Mapping[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    label = arrays["label"]
    if detector.get("mechanism_scores_available", True):
        detection = detection_summary(
            label,
            scores,
            arrays["source_id"],
            bootstrap=bootstrap,
            seed=seed,
        )
    else:
        detection = {
            name: {
                "auroc": None,
                "auprc": None,
                "auprc_lift": None,
                "auroc_ci95": [None, None],
                "auprc_ci95": [None, None],
                "unavailable_reason": detector.get("reason"),
            }
            for name in SCORE_ORDER
        }
    return {
        "schema": "ragtruth-mechanism-innovation-detection-v1",
        "task_type": task_type,
        "samples": int(np.unique(arrays["sample_id"]).size),
        "sources": int(np.unique(arrays["source_id"]).size),
        "tokens": int(len(label)),
        "hallucinated_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "primary_score": PRIMARY_SCORE,
        "control_scores": list(CONTROL_SCORES),
        "score_definitions": SCORE_DEFINITIONS,
        "score_direction": "higher is more hallucination-like; never label-flipped",
        "detection_estimand": "token_micro",
        "detection_bootstrap_unit": "source_id_cluster",
        "detection": detection,
        "detector": dict(detector),
        "labels_used_during": "posthoc_evaluation_only_after_score_freeze",
        "analysis_scope": "source-crossfit mechanism-state innovation by task",
    }


def evaluate_all(
    *,
    inputs: Iterable[tuple[str | Path, str | Path]],
    task_type: str,
    output: str | Path,
    bootstrap: int = 1000,
    seed: int = 20260828,
    position_bin: int = 16,
) -> dict[str, Any]:
    """Pool physical shards, freeze one OOF detector, and only then open labels."""

    task_type = canonical_task_type(task_type)
    records, manifests = _pool_records(inputs, task_type)
    detector_records = [
        {
            name: record[name]
            for name in (
                "sample_id",
                "source_id",
                "task_type",
                "path",
                "response_tokens",
                "physical_shard",
            )
            if name in record
        }
        for record in records
    ]
    per_sample, detector = score_records(detector_records, seed=seed)
    frozen, audit_names = _concatenate_label_free(records, per_sample)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = output.with_name("frozen_scores.npz")
    frozen_sha256 = _write_frozen(frozen_path, frozen)
    label = _load_labels(records, frozen)
    merged = {**frozen, "label": label}
    scores = {name: merged[name] for name in SCORE_ORDER}
    report = build_report(
        task_type=task_type,
        arrays=merged,
        scores=scores,
        detector=detector,
        bootstrap=bootstrap,
        seed=seed,
    )
    report["group_difference_audit"] = group_difference_audit(
        merged,
        audit_names,
        position_bin=position_bin,
        bootstrap=bootstrap,
        seed=seed + len(SCORE_ORDER),
    )
    scores_path = output.with_name("token_scores.npz")
    np.savez_compressed(scores_path, **merged)
    figures = output.parent / "figures"
    plot_population(
        merged["label"],
        scores,
        merged["token_index"],
        merged["response_length"],
        report,
        figures,
    )
    report.update(
        {
            "frozen_scores": str(frozen_path),
            "frozen_scores_sha256": frozen_sha256,
            "token_scores": str(scores_path),
            "figures": str(figures),
            "physical_cache_shards": len(manifests),
            "capture_complete": all(manifest["complete"] for manifest in manifests),
        }
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def plot_saved_sample(
    *,
    inputs: Iterable[str | Path],
    sample_id: str,
    model_path: str | Path,
    output: str | Path,
) -> dict[str, str]:
    """Render one saved mechanism state without replaying the model or labels."""

    from transformers import AutoTokenizer

    for trace_value in inputs:
        trace_root = Path(trace_value)
        _load_manifest(trace_root)
        row = next(
            (
                current
                for current in load_index(trace_root)
                if str(current["sample_id"]) == str(sample_id)
            ),
            None,
        )
        if row is None:
            continue
        artifact = _load_artifact(trace_root / "samples" / row["path"])
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        layers = layer_audit_metrics(artifact)
        layers.update(
            {
                "edge_history_share": layers["edge_response_history_share"],
                "edge_self_share": layers["edge_predictor_self_share"],
            }
        )
        trace = artifact["trace"]
        top_index = _array(trace["top_source_index"])
        top_mass = _array(trace["top_source_magnitude"])
        response_tokens = top_index.shape[1]
        source_flow = np.zeros(
            (response_tokens, len(artifact["token_ids"]) - 1), dtype=np.float32
        )
        response_index = np.broadcast_to(
            np.arange(response_tokens)[None, :, None], top_index.shape
        )
        valid = top_index >= 0
        np.add.at(
            source_flow,
            (response_index[valid], top_index[valid]),
            top_mass[valid],
        )
        edge_total = _array(trace["edge_role_energy"]).sum(axis=(0, 2, 3))
        source_flow /= edge_total[:, None] + _EPS
        shown = np.argsort(source_flow.sum(axis=0))[-16:][::-1]
        contrasts = factorial_contrasts(artifact)
        token_ids = _array(artifact["token_ids"])
        record = {
            "sample_id": str(sample_id),
            "token_text": tokenizer.convert_ids_to_tokens(
                token_ids[artifact["response_start"] :].tolist()
            ),
            "evidence_support": contrasts[:, 0],
            "history_support": contrasts[:, 1],
            "route_interaction": contrasts[:, 2],
            "source_token_text": [
                f"{index}:" + tokenizer.convert_ids_to_tokens(int(token_ids[index]))
                for index in shown
            ],
            "source_flow": source_flow[:, shown].T,
        }
        plot_sample_dashboard(record, layers, Path(output))
        return {"sample_id": str(sample_id), "output": str(output)}
    raise ValueError(f"sample {sample_id} was not found in the saved mechanism states")
