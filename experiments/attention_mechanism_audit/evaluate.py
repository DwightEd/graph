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

from .capture import (
    EVIDENCE,
    HISTORY,
    PATHWAY_CONTRAST_NAMES,
    PATHWAY_STAGE_NAMES,
    ROLE_NAMES,
)
from .collect import (
    SCHEMA,
    VERSION,
    load_index,
    target_response_sha256,
    token_ids_sha256,
    validate_saved_artifact,
)
from .data import canonical_task_type
from .detect import SCORE_DEFINITIONS, SCORE_NAMES, factorial_contrasts, score_records
from .graph import build_graph, route_contraction, route_mass_contraction
from .visualize import plot_population, plot_sample_dashboard

SCORE_ORDER = SCORE_NAMES
PRIMARY_SCORE = SCORE_NAMES[0]
CONTROL_SCORES = SCORE_NAMES[1:]
ROUTE_FIELDS = {
    "attention": "attention_role_mass",
    "edge": "edge_role_mass",
}
PATHWAY_ATTENTION = PATHWAY_STAGE_NAMES.index("attention")
POPULATION_LAYER_AUDITS = (
    *(
        f"edge_{role}_{statistic}"
        for role in ("evidence", "response_history")
        for statistic in (
            "mass_share",
            "effective_routes",
            "effective_rank",
            "head_entropy",
            "head_entropy_spread",
            "head_top1",
            "head_top1_spread",
        )
    ),
    "evidence_within_head_cancellation",
    "response_history_within_head_cancellation",
    "head_coherence_evidence",
    "head_coherence_response_history",
    *(
        f"attention_{role}_{statistic}"
        for role in ("evidence", "response_history")
        for statistic in (
            "mass_share",
            "effective_routes",
        )
    ),
    *(
        f"pathway_{contrast}_{statistic}"
        for contrast in PATHWAY_CONTRAST_NAMES
        for statistic in (
            "attention_norm",
            "mlp_projection",
            "output_gain",
            "output_cosine",
        )
    ),
    *(
        f"edge_{role}_{statistic}"
        for role in ("evidence", "response_history")
        for statistic in (
            "head_cover_size",
            "head_cover_size_spread",
            "anchor_persistence",
        )
    ),
)
ONSET_AUDITS = {
    "edge_evidence_mass_share_mean",
    "edge_response_history_mass_share_mean",
    "edge_evidence_mass_contraction",
    "edge_response_history_mass_contraction",
    "edge_evidence_head_cover_size_mean",
    "edge_response_history_head_cover_size_mean",
    "edge_evidence_head_cover_size_spread_mean",
    "edge_response_history_head_cover_size_spread_mean",
    "edge_evidence_anchor_persistence_mean",
    "edge_response_history_anchor_persistence_mean",
    *(
        f"edge_{role}_{statistic}_mean"
        for role in ("evidence", "response_history")
        for statistic in (
            "effective_routes",
            "effective_rank",
            "head_entropy",
            "head_top1",
        )
    ),
    *(
        f"attention_{role}_effective_routes_mean"
        for role in ("evidence", "response_history")
    ),
    "pathway_evidence_mlp_projection_mean",
    "causal_evidence_support",
    "causal_history_support",
    "causal_interaction",
    "unsupported_history_takeover_raw",
}
_EPS = 1e-12


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(
    path: str | Path, record: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    if record is not None:
        validate_saved_artifact(artifact, record)
    return artifact


def _layer_reductions(
    value: np.ndarray, name: str, valid: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Reduce a label-free ``[layer, token]`` measurement."""

    valid = np.ones(value.shape, dtype=bool) if valid is None else valid

    def mean(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        selected = valid[start:stop]
        total = selected.sum(axis=0)
        average = (value[start:stop] * selected).sum(axis=0) / np.maximum(total, 1)
        return average, total > 0

    width = max(value.shape[0] // 3, 1)
    average, _ = mean(0, len(value))
    early, early_valid = mean(0, width)
    late, late_valid = mean(len(value) - width, len(value))
    shift = np.where(early_valid & late_valid, late - early, 0.0)
    return {
        f"{name}_mean": average.astype(np.float32),
        f"{name}_layer_shift": shift.astype(np.float32),
    }


def _weighted_head_summary(
    value: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize heads without giving a negligible route equal weight."""

    total = mass.sum(axis=2)
    mean = (value * mass).sum(axis=2) / np.maximum(total, _EPS)
    variance = (np.square(value - mean[:, :, None, :]) * mass).sum(axis=2) / np.maximum(
        total, _EPS
    )
    return mean, np.sqrt(np.maximum(variance, 0.0))


def _anchor_persistence(anchor: np.ndarray, mass: np.ndarray) -> np.ndarray:
    persistence = np.zeros(anchor.shape[:2], dtype=np.float64)
    valid = (anchor[:, 1:] >= 0) & (anchor[:, :-1] >= 0)
    weight = mass[:, 1:] * valid
    persistence[:, 1:] = (weight * (anchor[:, 1:] == anchor[:, :-1])).sum(
        axis=2
    ) / np.maximum(weight.sum(axis=2), _EPS)
    return persistence


def layer_audit_metrics(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Reduce only the reporting copy of head-resolved route measurements."""

    trace = artifact["trace"]
    result: dict[str, np.ndarray] = {}
    for family, mass_field in ROUTE_FIELDS.items():
        mass = _array(trace[mass_field]).astype(np.float64)
        role_mass = mass.sum(axis=2)
        share = role_mass / np.maximum(role_mass.sum(axis=-1, keepdims=True), _EPS)
        effective_routes = _array(trace[f"{family}_role_effective_routes"]).astype(
            np.float64
        )
        effective_rank = _array(trace[f"{family}_role_effective_rank"]).astype(
            np.float64
        )
        entropy = _array(trace[f"{family}_role_source_entropy"]).astype(np.float64)
        top1 = _array(trace[f"{family}_role_top1"]).astype(np.float64)
        entropy_mean, entropy_spread = _weighted_head_summary(entropy, mass)
        top1_mean, top1_spread = _weighted_head_summary(top1, mass)
        for role_index, role in enumerate(ROLE_NAMES):
            prefix = f"{family}_{role}"
            result[f"{prefix}_mass_share"] = share[..., role_index]
            result[f"{prefix}_effective_routes"] = effective_routes[..., role_index]
            result[f"{prefix}_effective_rank"] = effective_rank[..., role_index]
            result[f"{prefix}_head_entropy"] = entropy_mean[..., role_index]
            result[f"{prefix}_head_entropy_spread"] = entropy_spread[..., role_index]
            result[f"{prefix}_head_top1"] = top1_mean[..., role_index]
            result[f"{prefix}_head_top1_spread"] = top1_spread[..., role_index]

    edge_head_mass = _array(trace["edge_role_mass"]).astype(np.float64)
    edge_mass = edge_head_mass.sum(axis=2)
    write_norm = _array(trace["head_role_write_norm"]).astype(np.float64).sum(axis=2)
    coherence = _array(trace["role_head_coherence"]).astype(np.float64)
    for role_index, role in enumerate(ROLE_NAMES):
        mass = edge_mass[..., role_index]
        cancellation = np.zeros_like(mass)
        valid = mass > _EPS
        cancellation[valid] = 1.0 - np.minimum(
            write_norm[..., role_index][valid] / mass[valid], 1.0
        )
        result[f"{role}_within_head_cancellation"] = cancellation
        result[f"head_coherence_{role}"] = coherence[..., role_index]

    route_roles = (EVIDENCE, HISTORY)
    cover_size = _array(trace["route_source_cover_size"]).astype(np.float64)
    cover_mean, cover_spread = _weighted_head_summary(
        cover_size, edge_head_mass[..., route_roles]
    )
    edge_anchor = _array(trace["edge_role_anchor_index"]).astype(np.int64)
    for route_index, role_index in enumerate(route_roles):
        role = ROLE_NAMES[role_index]
        result[f"edge_{role}_head_cover_size"] = cover_mean[..., route_index]
        result[f"edge_{role}_head_cover_size_spread"] = cover_spread[..., route_index]
        result[f"edge_{role}_anchor_persistence"] = _anchor_persistence(
            edge_anchor[..., role_index], edge_head_mass[..., role_index]
        )

    effect_norm = _array(trace["pathway_effect_norm"]).astype(np.float64)
    projection = _array(trace["pathway_mlp_projection"]).astype(np.float64)
    output_gain = _array(trace["pathway_pre_output_gain"]).astype(np.float64)
    output_cosine = _array(trace["pathway_pre_output_cosine"]).astype(np.float64)
    pathway_valid = _array(trace["pathway_valid"]).astype(bool)
    cosine_valid = _array(trace["pathway_cosine_valid"]).astype(bool)
    for contrast_index, contrast in enumerate(PATHWAY_CONTRAST_NAMES):
        prefix = f"pathway_{contrast}"
        result[f"{prefix}_attention_norm"] = effect_norm[
            ..., contrast_index, PATHWAY_ATTENTION
        ]
        result[f"{prefix}_mlp_projection"] = projection[..., contrast_index]
        result[f"{prefix}_output_gain"] = output_gain[..., contrast_index]
        result[f"{prefix}_output_cosine"] = output_cosine[..., contrast_index]
        result[f"{prefix}_valid"] = pathway_valid[..., contrast_index]
        result[f"{prefix}_cosine_valid"] = cosine_valid[..., contrast_index]
    return result


def token_audit_metrics(artifact: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build fixed route, pathway, and endpoint diagnostics before labels open."""

    metrics: dict[str, np.ndarray] = {}
    layers = layer_audit_metrics(artifact)
    for name in POPULATION_LAYER_AUDITS:
        valid = None
        if name.startswith("pathway_") and name.endswith("_output_cosine"):
            contrast = name.split("_", 2)[1]
            valid = layers[f"pathway_{contrast}_cosine_valid"]
        elif name.startswith("pathway_") and name.endswith(
            ("_mlp_projection", "_output_gain")
        ):
            contrast = name.split("_", 2)[1]
            valid = layers[f"pathway_{contrast}_valid"]
        metrics.update(_layer_reductions(layers[name], name, valid))
    for contrast in PATHWAY_CONTRAST_NAMES:
        metrics[f"pathway_{contrast}_valid_mean"] = (
            layers[f"pathway_{contrast}_valid"]
            .mean(axis=0, dtype=np.float64)
            .astype(np.float32)
        )
        metrics[f"pathway_{contrast}_cosine_valid_mean"] = (
            layers[f"pathway_{contrast}_cosine_valid"]
            .mean(axis=0, dtype=np.float64)
            .astype(np.float32)
        )
    for role in ("evidence", "response_history"):
        route_value, route_valid = route_contraction(
            artifact, role=role, return_valid=True
        )
        mass_value, mass_valid = route_mass_contraction(
            artifact, role=role, return_valid=True
        )
        route_name = f"edge_{role}_route_contraction"
        mass_name = f"edge_{role}_mass_contraction"
        metrics[route_name] = route_value.astype(np.float32)
        metrics[f"{route_name}__valid"] = route_valid
        metrics[mass_name] = mass_value.astype(np.float32)
        metrics[f"{mass_name}__valid"] = mass_valid
    contrasts = factorial_contrasts(artifact).astype(np.float32)
    inputs = artifact["score_inputs"]
    full = _array(inputs["full_logprob"]).astype(np.float32)
    no_evidence = _array(inputs["no_evidence_logprob"]).astype(np.float32)
    no_neither = _array(inputs["no_evidence_history_logprob"]).astype(np.float32)
    direct_evidence_with_history = full - no_evidence
    history_under_evidence_cut = no_evidence - no_neither
    metrics.update(
        {
            "causal_evidence_support": contrasts[:, 0],
            "causal_history_support": contrasts[:, 1],
            "causal_interaction": contrasts[:, 2],
            "direct_evidence_support_with_history": direct_evidence_with_history,
            "history_support_under_direct_evidence_cut": history_under_evidence_cut,
            "unsupported_history_takeover_raw": (
                history_under_evidence_cut - direct_evidence_with_history
            ),
        }
    )
    return metrics


def _binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prevalence = float(label.mean())
    average_precision = float(average_precision_score(label, score))
    return {
        "auroc": float(roc_auc_score(label, score)),
        "average_precision": average_precision,
        "ap_lift": average_precision / prevalence,
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
            "average_precision_low": None,
            "average_precision_high": None,
        }
    return {
        "replicates": len(values),
        "auroc_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_high": float(np.quantile(values[:, 0], 0.975)),
        "average_precision_low": float(np.quantile(values[:, 1], 0.025)),
        "average_precision_high": float(np.quantile(values[:, 1], 0.975)),
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
                "average_precision": None,
                "ap_lift": None,
                "auroc_ci95": [None, None],
                "average_precision_ci95": [None, None],
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
                    "average_precision_ci95": [
                        interval["average_precision_low"],
                        interval["average_precision_high"],
                    ],
                    "bootstrap_replicates": interval["replicates"],
                }
            )
        else:
            result.update(
                {
                    "auroc_ci95": [None, None],
                    "average_precision_ci95": [None, None],
                }
            )
        results[name] = result
    return results


def _position_match_design(
    label: np.ndarray,
    sample_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
    eligible: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    relative = np.minimum(
        ((token_index + 0.5) * 10 / response_length).astype(np.int16), 9
    )
    absolute = token_index // position_bin
    cells: dict[tuple[str, int, int], list[int]] = {}
    for index, key in enumerate(zip(sample_id, absolute, relative)):
        cells.setdefault(key, []).append(index)
    matched = []
    eligible = np.ones(len(label), dtype=bool) if eligible is None else eligible
    for (sample, _absolute, _relative), rows in cells.items():
        rows = np.asarray(rows)
        rows = rows[eligible[rows]]
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
    return float(values.mean()), interval, len(values)


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
        "matched_samples": len(by_sample),
        "matched_cells": len(matched),
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
    eligible: np.ndarray | None = None,
) -> dict[str, object]:
    """Compare route change at hallucination onset with local correct pivots."""

    by_source: dict[str, list[float]] = {}
    events = controls = 0
    eligible = np.ones(len(label), dtype=bool) if eligible is None else eligible
    for sample in np.unique(sample_id):
        rows = np.flatnonzero(sample_id == sample)
        rows = rows[eligible[rows]]
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


def _has_onset_audit(name: str) -> bool:
    return name in ONSET_AUDITS


def _requires_strict_history(name: str) -> bool:
    return "history" in name or "interaction" in name


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
    all_tokens = np.ones(len(label), dtype=bool)
    detection_valid = np.asarray(arrays["detection_valid"], dtype=bool)
    matched = _position_match_design(
        label,
        arrays["sample_id"],
        arrays["token_index"],
        arrays["response_length"],
        position_bin=position_bin,
        eligible=all_tokens,
    )
    history_matched = _position_match_design(
        label,
        arrays["sample_id"],
        arrays["token_index"],
        arrays["response_length"],
        position_bin=position_bin,
        eligible=detection_valid,
    )
    sample_source = {
        str(sample): str(source)
        for sample, source in zip(arrays["sample_id"], arrays["source_id"])
    }
    metrics = {}
    onset = {}
    for offset, name in enumerate(audit_names):
        value = arrays[name]
        history_scope = _requires_strict_history(name)
        eligible = (detection_valid if history_scope else all_tokens).copy()
        validity_name = f"{name}__valid"
        if validity_name in arrays:
            eligible &= np.asarray(arrays[validity_name], dtype=bool)
        base_matched = history_matched if history_scope else matched
        current_matched = []
        for sample, positive, negative, _weight in base_matched:
            positive = positive[eligible[positive]]
            negative = negative[eligible[negative]]
            if len(positive) and len(negative):
                weight = len(positive) * len(negative) / (len(positive) + len(negative))
                current_matched.append((sample, positive, negative, float(weight)))
        correct = eligible & ~label
        hallucinated = eligible & label
        metrics[name] = {
            "token_scope": (
                "valid_comparable_tokens"
                if validity_name in arrays
                else (
                    "strict_history_eligible"
                    if history_scope
                    else "all_response_tokens"
                )
            ),
            "valid_tokens": int(eligible.sum()),
            "correct_mean": (float(value[correct].mean()) if correct.any() else None),
            "hallucinated_mean": (
                float(value[hallucinated].mean()) if hallucinated.any() else None
            ),
            **_position_matched_difference(
                value,
                current_matched,
                sample_source,
                bootstrap=bootstrap,
                seed=seed + offset,
            ),
        }
        if _has_onset_audit(name):
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
                eligible=eligible,
            )
    covered = sum(len(positive) for _sample, positive, _negative, _weight in matched)
    history_covered = sum(
        len(positive) for _sample, positive, _negative, _weight in history_matched
    )
    eligible_hallucinations = int((detection_valid & label).sum())
    return {
        "role": "posthoc_mechanism_audit_not_score_selection",
        "token_scope": "all_response_tokens_including_pre_history_prefix",
        "matching": "sample_id + absolute_position_bin + relative_position_decile",
        "aggregation": "matched cells -> response -> equal source",
        "bootstrap_unit": "source_id",
        "position_bin": int(position_bin),
        "strict_history_eligible_tokens": int(detection_valid.sum()),
        "strict_history_eligible_hallucinated_tokens": int(eligible_hallucinations),
        "strict_history_covered_hallucinated_tokens": int(history_covered),
        "strict_history_hallucinated_token_coverage": float(
            history_covered / max(eligible_hallucinations, 1)
        ),
        "covered_hallucinated_tokens": int(covered),
        "hallucinated_token_coverage": float(covered / max(label.sum(), 1)),
        "metrics": metrics,
        "onset_difference_in_difference": onset,
    }


def _load_manifest(trace_root: Path, task_type: str | None = None) -> dict:
    manifest = json.loads((trace_root / "manifest.json").read_text(encoding="utf-8"))
    required = (
        "observer_identity",
        "model_dtype",
        "capture_spec",
        "source_identity",
        "split_identity",
    )
    valid = (
        manifest.get("schema") == SCHEMA
        and manifest.get("version") == VERSION
        and manifest.get("labels_used") is False
        and all(name in manifest for name in required)
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
    shared_identity = None
    for shard, (trace_value, split_value) in enumerate(inputs):
        trace_root, split_root = Path(trace_value), Path(split_value)
        manifest = _load_manifest(trace_root, task_type)
        current_identity = {
            name: manifest[name]
            for name in (
                "schema",
                "version",
                "observer_identity",
                "model_dtype",
                "capture_spec",
                "source_identity",
                "labels_used",
            )
        }
        if shared_identity is None:
            shared_identity = current_identity
        elif current_identity != shared_identity:
            raise ValueError(
                "mechanism-state shards have different scientific identity"
            )
        manifests.append(manifest)
        artifact_contract = {
            "schema": manifest["schema"],
            "version": manifest["version"],
            "capture_spec": manifest["capture_spec"],
        }
        current = []
        for row in load_index(trace_root):
            if canonical_task_type(row["task_type"]) != task_type:
                continue
            if row.get("artifact_contract") != artifact_contract:
                raise ValueError("index row does not match its shard capture contract")
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
        "detection_valid": [],
        **{name: [] for name in SCORE_ORDER},
    }
    audit_schema: tuple[str, ...] | None = None
    audit_names: tuple[str, ...] | None = None
    for record in records:
        sample = str(record["sample_id"])
        if sample not in scores:
            raise ValueError(f"detector did not score sample {sample}")
        current_scores = scores[sample]
        count = len(current_scores[PRIMARY_SCORE])
        if any(len(current_scores[name]) != count for name in SCORE_ORDER):
            raise ValueError(f"detector score length mismatch for sample {sample}")
        detection_valid = np.asarray(current_scores["detection_valid"], dtype=bool)
        if detection_valid.shape != (count,):
            raise ValueError(f"detector validity mismatch for sample {sample}")
        expected = record.get("response_tokens")
        if expected is not None and int(expected) != count:
            raise ValueError(f"artifact/index length mismatch for sample {sample}")
        audit = token_audit_metrics(_load_artifact(record["path"], record))
        if any(len(value) != count for value in audit.values()):
            raise ValueError(f"audit length mismatch for sample {sample}")
        if audit_schema is None:
            audit_schema = tuple(audit)
            audit_names = tuple(
                name for name in audit_schema if not name.endswith("__valid")
            )
            output.update({name: [] for name in audit_schema})
        elif tuple(audit) != audit_schema:
            raise ValueError("audit measurement schema changed between samples")
        output["sample_id"].append(np.repeat(sample, count))
        output["source_id"].append(np.repeat(str(record["source_id"]), count))
        output["physical_shard"].append(
            np.full(count, int(record["physical_shard"]), dtype=np.int8)
        )
        output["token_index"].append(np.arange(count, dtype=np.int32))
        output["response_length"].append(np.full(count, count, dtype=np.int32))
        output["detection_valid"].append(detection_valid)
        for name in SCORE_ORDER:
            output[name].append(np.asarray(current_scores[name], dtype=np.float32))
        for name, value in audit.items():
            dtype = bool if name.endswith("__valid") else np.float32
            output[name].append(np.asarray(value, dtype=dtype))
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


def _validate_sample_identity(record: Mapping[str, Any], sample: Any) -> Any:
    """Match the frozen trace index to the exact cache sample being labelled."""

    if str(sample.source_id) != str(record["source_id"]):
        raise ValueError("source_id changed between mechanism state and labels")
    if canonical_task_type(sample.task_type) != canonical_task_type(
        record["task_type"]
    ):
        raise ValueError("task_type changed between mechanism state and labels")
    expected_generator = record.get("generator_model")
    if (
        expected_generator is not None
        and sample.generator_model is not None
        and str(sample.generator_model) != str(expected_generator)
    ):
        raise ValueError("generator_model changed between mechanism state and labels")
    attention = sample.attention()
    expected_prompt = record.get("prompt_tokens")
    if expected_prompt is not None and int(attention.response_idx) != int(
        expected_prompt
    ):
        raise ValueError(
            "prompt token count changed between mechanism state and labels"
        )
    expected_digest = record.get("target_response_sha256")
    if expected_digest is not None and target_response_sha256(
        attention.token_ids, int(attention.response_idx)
    ) != str(expected_digest):
        raise ValueError(
            "response token IDs changed between mechanism state and labels"
        )
    expected_all_tokens = record.get("token_ids_sha256")
    if expected_all_tokens is not None and token_ids_sha256(attention.token_ids) != str(
        expected_all_tokens
    ):
        raise ValueError(
            "prompt or response token IDs changed between state and labels"
        )
    return attention


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
        count = int(frozen["response_length"][offset])
        try:
            attention = _validate_sample_identity(record, sample)
            if len(attention.token_ids) - int(attention.response_idx) != count:
                raise ValueError(
                    "response token count changed between mechanism state and labels"
                )
            token_label = prepared.response_labels(sample).cpu().numpy().astype(bool)
        finally:
            sample.release_attention()
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
    valid = np.asarray(arrays["detection_valid"], dtype=bool)
    evaluated_label = label[valid]
    evaluated_scores = {name: score[valid] for name, score in scores.items()}
    detection = detection_summary(
        evaluated_label,
        evaluated_scores,
        arrays["source_id"][valid],
        bootstrap=bootstrap,
        seed=seed,
    )
    if not detector.get("mechanism_scores_available", True):
        detection.update(
            {
                name: {
                    "auroc": None,
                    "average_precision": None,
                    "ap_lift": None,
                    "auroc_ci95": [None, None],
                    "average_precision_ci95": [None, None],
                    "unavailable_reason": detector.get("reason"),
                }
                for name in SCORE_ORDER[:-1]
            }
        )
    return {
        "schema": "ragtruth-route-adoption-detection-v1",
        "task_type": task_type,
        "samples": int(np.unique(arrays["sample_id"]).size),
        "sources": int(np.unique(arrays["source_id"]).size),
        "tokens": len(label),
        "hallucinated_tokens": int(label.sum()),
        "total_prevalence": float(label.mean()),
        "evaluated_tokens": int(valid.sum()),
        "evaluated_positives": int(evaluated_label.sum()),
        "evaluated_samples": int(np.unique(arrays["sample_id"][valid]).size),
        "evaluated_sources": int(np.unique(arrays["source_id"][valid]).size),
        "prevalence": (float(evaluated_label.mean()) if len(evaluated_label) else None),
        "primary_score": PRIMARY_SCORE,
        "control_scores": list(CONTROL_SCORES),
        "score_definitions": SCORE_DEFINITIONS,
        "score_direction": "higher is more hallucination-like; never label-flipped",
        "detection_estimand": "token_micro",
        "detection_token_scope": (
            "strict_history_eligible_response_tokens_token_index_ge_2"
        ),
        "detection_bootstrap_unit": "source_id_cluster",
        "detection": detection,
        "detector": dict(detector),
        "labels_used_during": "posthoc_evaluation_only_after_score_freeze",
        "analysis_scope": (
            "source-crossfit unsupported-history takeover with head-resolved "
            "route and attention/MLP pathway audit by task"
        ),
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
                "artifact_contract",
                "token_ids_sha256",
                "evidence_mask_sha256",
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
        merged["label"][merged["detection_valid"]],
        {name: score[merged["detection_valid"]] for name, score in scores.items()},
        merged["token_index"][merged["detection_valid"]],
        merged["response_length"][merged["detection_valid"]],
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
        artifact = _load_artifact(trace_root / "samples" / row["path"], row)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        layers = layer_audit_metrics(artifact)
        trace = artifact["trace"]
        layers["edge_evidence_head_entropy"] = _array(
            trace["edge_role_source_entropy"]
        )[..., EVIDENCE]
        layers["edge_history_head_entropy"] = _array(trace["edge_role_source_entropy"])[
            ..., HISTORY
        ]
        projection = _array(trace["pathway_mlp_projection"]).astype(np.float64)
        projection[~_array(trace["pathway_valid"]).astype(bool)] = np.nan
        layers["pathway_mlp_projection"] = projection
        contrasts = factorial_contrasts(artifact)
        token_ids = _array(artifact["token_ids"])
        response_start = int(artifact["response_start"])
        record = {
            "sample_id": str(sample_id),
            "token_text": tokenizer.convert_ids_to_tokens(
                token_ids[response_start:].tolist()
            ),
            "predictor_position": np.arange(response_start - 1, len(token_ids) - 1),
            "evidence_support": contrasts[:, 0],
            "history_support": contrasts[:, 1],
            "route_interaction": contrasts[:, 2],
            "evidence_route_contraction": route_contraction(
                artifact, role="evidence", family="edge"
            ),
            "history_route_contraction": route_contraction(
                artifact, role="response_history", family="edge"
            ),
        }
        plot_sample_dashboard(record, layers, build_graph(artifact), Path(output))
        return {"sample_id": str(sample_id), "output": str(output)}
    raise ValueError(f"sample {sample_id} was not found in the saved mechanism states")
