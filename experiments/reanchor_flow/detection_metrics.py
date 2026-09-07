"""Token detection metrics with paired, source-cluster uncertainty estimates.

Scores always increase with hallucination risk. Labels only enter this reporting
module; they do not determine score direction or detector parameters. AUPRC here
means average precision (the stepwise precision-recall integral).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


class RankedScores:
    """Sort once, then evaluate exact tied-score metrics for source weights."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray, source: np.ndarray):
        order = np.argsort(-scores, kind="stable")
        self.labels = np.asarray(labels[order], dtype=np.float64)
        self.source = source[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]

    def counts(self, source_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = source_weights[self.source]
        positive = np.add.reduceat(weights * self.labels, self.starts)
        total = np.add.reduceat(weights, self.starts)
        return positive, total - positive

    def evaluate(self, source_weights: np.ndarray) -> np.ndarray:
        positive, negative = self.counts(source_weights)
        p, n = positive.sum(), negative.sum()
        if p == 0:
            return np.array([np.nan, np.nan])
        cumulative_p = np.cumsum(positive)
        cumulative_n = np.cumsum(negative)
        total = cumulative_p + cumulative_n
        precision = np.divide(cumulative_p, total, out=np.zeros_like(total), where=total > 0)
        ap = np.dot(positive, precision) / p
        auc = np.dot(positive, n - cumulative_n + 0.5 * negative) / (p * n) if n else np.nan
        return np.array([auc, ap])


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _interval(values: np.ndarray) -> list[float] | None:
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def _validate(labels, scores, source_ids, task_types):
    labels = np.asarray(labels)
    sources = np.asarray(source_ids, dtype=str)
    tasks = np.asarray(task_types, dtype=str)
    if labels.ndim != 1 or sources.shape != labels.shape or tasks.shape != labels.shape:
        raise ValueError("labels, source_ids and task_types must be aligned one-dimensional arrays")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("labels must be 0, 1, or -1 for unknown")
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in scores.items()}
    if not arrays:
        raise ValueError("at least one score is required")
    for name, value in arrays.items():
        if value.shape != labels.shape or not np.isfinite(value).all():
            raise ValueError(f"score {name!r} must be finite and aligned with labels")
    return labels, arrays, sources, tasks


def _group_report(labels, scores, sources, *, primary, bootstrap, rng, group_name):
    known = labels >= 0
    known_labels = labels[known]
    unique_sources, source = np.unique(sources[known], return_inverse=True)
    count = len(unique_sources)
    result = {
        "tokens": len(labels),
        "known_tokens": int(known.sum()),
        "unknown_tokens": int((~known).sum()),
        "sources": int(count),
        "positives": int(known_labels.sum()),
        "prevalence": float(known_labels.mean()) if known.any() else None,
        "scores": {},
        "paired_differences": {},
    }
    names = list(scores)
    point = np.full((len(names), 2), np.nan)
    draws = np.full((bootstrap, len(names), 2), np.nan)
    if count:
        ranked = [RankedScores(known_labels, scores[name][known], source) for name in names]
        ones = np.ones(count)
        point = np.stack([item.evaluate(ones) for item in ranked])
        for b in tqdm(range(bootstrap), desc=f"source bootstrap {group_name}", unit="draw", leave=False):
            # Every token from a resampled source receives the same multiplicity.
            # All scorers share this draw, so differences are paired.
            weights = rng.multinomial(count, np.full(count, 1.0 / count))
            draws[b] = np.stack([item.evaluate(weights) for item in ranked])
    for i, name in enumerate(names):
        result["scores"][name] = {
            "auroc": _number(point[i, 0]),
            "auprc": _number(point[i, 1]),
            "auroc_ci95": _interval(draws[:, i, 0]),
            "auprc_ci95": _interval(draws[:, i, 1]),
            "bootstrap_valid": {metric: int(np.isfinite(draws[:, i, m]).sum())
                                for m, metric in enumerate(("auroc", "auprc"))},
        }
    main = names.index(primary)
    for i, name in enumerate(names):
        if i == main:
            continue
        difference = point[main] - point[i]
        paired = draws[:, main] - draws[:, i]
        result["paired_differences"][name] = {
            "auroc": _number(difference[0]),
            "auprc": _number(difference[1]),
            "auroc_ci95": _interval(paired[:, 0]),
            "auprc_ci95": _interval(paired[:, 1]),
            "bootstrap_valid": {metric: int(np.isfinite(paired[:, m]).sum())
                                for m, metric in enumerate(("auroc", "auprc"))},
        }
    return result


def detection_report(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    source_ids: np.ndarray,
    task_types: np.ndarray,
    *,
    primary: str = "routing_joint",
    bootstrap: int = 200,
    seed: int = 2026,
) -> dict:
    """Evaluate every annotated token and bootstrap complete source clusters.

    Confidence intervals describe source-sampling uncertainty for the frozen
    detector. They do not include model-refitting uncertainty. The pooled ALL
    estimate is token-weighted; task-specific estimates expose task mixtures.
    """
    labels, scores, sources, tasks = _validate(labels, scores, source_ids, task_types)
    if primary not in scores:
        raise ValueError(f"primary score {primary!r} is missing")
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    rng = np.random.default_rng(seed)
    groups = {}
    for task in ["ALL", *sorted(set(tasks))]:
        mask = np.ones(len(labels), dtype=bool) if task == "ALL" else tasks == task
        groups[task] = _group_report(
            labels[mask], {name: value[mask] for name, value in scores.items()}, sources[mask],
            primary=primary, bootstrap=bootstrap, rng=rng, group_name=task,
        )
    return {
        "primary": primary,
        "score_direction": "higher_is_more_hallucinated",
        "auprc_definition": "average_precision",
        "aggregation": "token_weighted",
        "bootstrap": {"unit": "source_id", "replicates": bootstrap, "seed": seed,
                      "interval": "percentile_95", "refits_detector": False},
        "groups": groups,
    }


def render_detection_report(
    path: str | Path, labels: np.ndarray, scores: dict[str, np.ndarray],
    task_types: np.ndarray, *, primary: str = "routing_joint",
) -> None:
    """Render exact-tie ROC and precision-recall curves for ALL and each task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, scores, _, tasks = _validate(labels, scores, np.arange(len(labels)), task_types)
    groups = ["ALL", *sorted(set(tasks))]
    figure, axes = plt.subplots(len(groups), 2, figsize=(12, 4 * len(groups)), squeeze=False)
    for row, task in enumerate(groups):
        mask = (labels >= 0) & ((tasks == task) if task != "ALL" else True)
        y = labels[mask]
        for name, values in scores.items():
            if not len(y):
                continue
            ranked = RankedScores(y, values[mask], np.zeros(len(y), dtype=int))
            positive, negative = ranked.counts(np.ones(1))
            p, n = positive.sum(), negative.sum()
            if not p:
                continue
            tp, fp = np.cumsum(positive), np.cumsum(negative)
            recall, precision = tp / p, tp / (tp + fp)
            width = 2.3 if name == primary else 1.3
            auc, ap = ranked.evaluate(np.ones(1))
            if n:
                axes[row, 0].plot(np.r_[0, fp / n], np.r_[0, recall], linewidth=width,
                                  label=f"{name}: {auc:.3f}")
            axes[row, 1].step(np.r_[0, recall], np.r_[precision[0], precision], where="pre",
                              linewidth=width, label=f"{name}: AP {ap:.3f}")
        axes[row, 0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
        if len(y):
            axes[row, 1].axhline(y.mean(), color="0.6", linestyle="--", linewidth=1,
                                label=f"prevalence: {y.mean():.3f}")
        for col, (xlabel, ylabel) in enumerate((("False positive rate", "True positive rate"),
                                               ("Recall", "Precision"))):
            axis = axes[row, col]
            axis.set(title=f"{task} — {'ROC' if col == 0 else 'precision-recall'}",
                     xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
            axis.grid(alpha=0.2)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
