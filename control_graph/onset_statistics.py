"""Source-balanced summaries for matched factual-onset events."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from control_graph.metrics import binary_detection_metrics

PAIRED_METRICS = (
    "instability",
    "predictor_entropy",
    "remote_gain",
    "evidence_gain",
    "far_history_gain",
    "local_loss",
)
MODES = ("focused_evidence", "focused_far_relay", "diffuse_global", "local_persistence")


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
) -> dict:
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
        f"{role}_instability_vs_{metric}": _source_correlation(
            pairs, sources, role, metric, bootstrap, seed
        )
        for role in ("onset", "control")
        for metric in ("remote_gain", "predictor_entropy")
    }
    return {
        "schema": "control-graph/onset-choice-audit@1",
        "samples_processed": processed,
        "samples_skipped": skipped,
        "hallucination_spans": spans,
        "content_onsets": content_onsets,
        "matched_pairs": len(pairs),
        "unmatched_onsets": content_onsets - len(pairs),
        "sources": len(set(sources)),
        "settings": {
            "pre_window": pre_window,
            "match_window": match_window,
            "bootstrap": bootstrap,
            "matching": "same_response_nearest_same_token_class_with_normal_window",
        },
        "classification": classification,
        "paired_differences": differences,
        "joint": joint,
        "lookback_modes": {
            role: _mode_distribution(pairs, sources, role)
            for role in ("onset", "control")
        },
        "events": str(event_path),
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


def _source_correlation(
    pairs: list[dict],
    sources: np.ndarray,
    role: str,
    metric: str,
    bootstrap: int,
    seed: int,
) -> dict:
    count = len(np.unique(sources))
    instability = _source_means(
        np.asarray([pair[role]["instability"] for pair in pairs]), sources
    )
    comparison = _source_means(
        np.asarray([pair[role][metric] for pair in pairs]), sources
    )
    estimate = _spearman(instability, comparison)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap):
        selected = rng.integers(0, count, count)
        value = _spearman(instability[selected], comparison[selected])
        if value is not None:
            estimates.append(value)
    return {
        "estimate": estimate,
        "confidence_interval": np.quantile(estimates, [0.025, 0.975]).tolist()
        if estimates
        else None,
        "sources": count,
        "valid_bootstrap_replicates": len(estimates),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank, right_rank = _ranks(left), _ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
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
