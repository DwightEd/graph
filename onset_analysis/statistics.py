"""Source-balanced summaries for matched factual-onset events."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from route_graph.metrics import binary_detection_metrics

PAIRED_METRICS = (
    "instability",
    "predictor_entropy",
    "remote_gain",
    "evidence_gain",
    "far_history_gain",
    "local_loss",
)
MODES = (
    "focused_evidence",
    "focused_far_history",
    "diffuse_global",
    "local_persistence",
)


def summarize_onsets(
    pairs: list[dict],
    *,
    processed: int,
    skipped: int,
    spans: int,
    content_onsets: int,
    pre_window: int,
    match_window: int,
    bootstrap: int,
    seed: int,
    event_path: Path,
    coverage: dict,
    hallucination_tokens: int,
) -> dict:
    summary = _summarize_pairs(pairs, bootstrap, seed)
    cohorts = {
        name: _summarize_pairs(
            [pair for pair in pairs if pair["is_first_error"] == first], bootstrap, seed
        )
        for name, first in (("first_error", True), ("later_onsets", False))
    }
    return {
        "schema": "onset-analysis/summary@1",
        "samples_processed": processed,
        "samples_skipped": skipped,
        "hallucination_spans": spans,
        "content_onsets": content_onsets,
        "unmatched_onsets": content_onsets - len(pairs),
        "coverage": {
            **coverage,
            "matched_first_errors": cohorts["first_error"]["matched_pairs"],
        },
        "span_structure": {
            "hallucination_tokens": hallucination_tokens,
            "continuation_tokens": hallucination_tokens - spans,
            "continuation_fraction": (hallucination_tokens - spans)
            / hallucination_tokens,
        },
        "settings": {
            "pre_window": pre_window,
            "match_window": match_window,
            "bootstrap": bootstrap,
            "matching": "same_response_nearest_same_token_class_with_normal_window",
        },
        "interpretation": {
            "margin": "observer_token_compatibility",
            "event": "first_alphanumeric_token_in_annotated_span_not_entity_recognition",
            "lookback": "group_mass_change_proxy_not_verified_reanchor_or_relay",
            "joint": "source_balanced_event_rank_correlation_with_source_bootstrap",
            "scope": "retrospective_matched_analysis_not_online_detection",
        },
        **summary,
        "cohorts": cohorts,
        "events": str(event_path),
    }


def _summarize_pairs(pairs: list[dict], bootstrap: int, seed: int) -> dict:
    if not pairs:
        return {
            "matched_pairs": 0,
            "sources": 0,
            "classification": None,
            "paired_differences": None,
            "joint": None,
            "lookback_modes": None,
        }
    sources = np.asarray([pair["source_id"] for pair in pairs], dtype=str)
    classification = {
        metric: _classification(pairs, sources, metric, bootstrap, seed)
        for metric in ("instability", "predictor_entropy", "remote_gain")
    }
    differences = {
        metric: _mean_difference(pairs, sources, metric, bootstrap, seed)
        for metric in PAIRED_METRICS
    }
    joint = {
        f"{role}_instability_vs_{metric}": _event_correlation(
            pairs, sources, role, metric, bootstrap, seed
        )
        for role in ("onset", "control")
        for metric in ("remote_gain", "predictor_entropy")
    }
    return {
        "matched_pairs": len(pairs),
        "sources": len(set(sources)),
        "classification": classification,
        "paired_differences": differences,
        "joint": joint,
        "lookback_modes": {
            role: _mode_distribution(pairs, sources, role)
            for role in ("onset", "control")
        },
    }


def _classification(
    pairs: list[dict], sources: np.ndarray, metric: str, bootstrap: int, seed: int
) -> dict:
    labels = np.tile([1, 0], len(pairs))
    values = np.asarray(
        [
            value
            for pair in pairs
            for value in (pair["onset"][metric], pair["control"][metric])
        ]
    )
    return binary_detection_metrics(
        labels,
        values,
        np.repeat(sources, 2),
        bootstrap=bootstrap,
        seed=seed,
        source_balanced=True,
    )


def _source_means(values: np.ndarray, sources: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[sources == source].mean() for source in np.unique(sources)]
    )


def _mean_difference(
    pairs: list[dict], sources: np.ndarray, metric: str, bootstrap: int, seed: int
) -> dict:
    differences = np.asarray(
        [pair["onset"][metric] - pair["control"][metric] for pair in pairs]
    )
    values = _source_means(differences, sources)
    rng = np.random.default_rng(seed)
    estimates = [
        float(rng.choice(values, len(values), replace=True).mean())
        for _ in range(bootstrap)
    ]
    return {
        "mean": float(values.mean()),
        "confidence_interval": np.quantile(estimates, [0.025, 0.975]).tolist()
        if estimates
        else None,
        "sources": len(values),
    }


def _event_correlation(
    pairs: list[dict],
    sources: np.ndarray,
    role: str,
    metric: str,
    bootstrap: int,
    seed: int,
) -> dict:
    groups, inverse, counts = np.unique(
        sources, return_inverse=True, return_counts=True
    )
    by_source = [np.flatnonzero(sources == source) for source in groups]
    weights = 1.0 / counts[inverse]
    instability = np.asarray([pair[role]["instability"] for pair in pairs])
    comparison = np.asarray([pair[role][metric] for pair in pairs])
    estimate = _spearman(instability, comparison, weights)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap if len(groups) > 1 else 0):
        selected = np.concatenate(
            [by_source[index] for index in rng.integers(0, len(groups), len(groups))]
        )
        value = _spearman(
            instability[selected], comparison[selected], weights[selected]
        )
        if value is not None:
            estimates.append(value)
    return {
        "estimate": estimate,
        "confidence_interval": np.quantile(estimates, [0.025, 0.975]).tolist()
        if estimates
        else None,
        "sources": len(groups),
        "events": len(pairs),
        "valid_bootstrap_replicates": len(estimates),
    }


def _spearman(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float | None:
    left_rank, right_rank = _ranks(left, weights), _ranks(right, weights)
    left_rank -= np.average(left_rank, weights=weights)
    right_rank -= np.average(right_rank, weights=weights)
    variance = np.average(left_rank**2, weights=weights) * np.average(
        right_rank**2, weights=weights
    )
    if variance == 0:
        return None
    return float(
        np.clip(
            np.average(left_rank * right_rank, weights=weights) / np.sqrt(variance),
            -1,
            1,
        )
    )


def _ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted midranks give each source equal total mass, including tied values."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    cumulative = 0.0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        mass = weights[order[start:stop]].sum()
        ranks[order[start:stop]] = cumulative + mass / 2
        cumulative += mass
        start = stop
    return ranks


def _mode_distribution(pairs: list[dict], sources: np.ndarray, role: str) -> dict:
    counts = {
        mode: sum(pair[role]["lookback_mode"] == mode for pair in pairs)
        for mode in MODES
    }
    balanced = {}
    for mode in MODES:
        indicators = np.asarray(
            [pair[role]["lookback_mode"] == mode for pair in pairs], dtype=float
        )
        balanced[mode] = float(_source_means(indicators, sources).mean())
    return {
        "event_counts": counts,
        "source_balanced_fractions": balanced,
    }
