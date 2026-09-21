"""Exact AUROC gain accounting by positive-token age; not causal attribution."""

import numpy as np

from ..evaluate import Ranking
from ..fixed_graph.evaluation import metric_arrays

PHASES = ("onset", "offset1_3", "offset4_7", "offset8_plus")


def token_phases(blocks):
    """Age resets at each mapped annotation, including adjacent annotations."""
    phases = []
    for block in blocks:
        values = np.full(block["tokens"], -1, dtype=int)
        for start, end in block["gold"]:
            age = np.arange(end - start)
            values[start:end] = np.searchsorted([1, 4, 8], age, side="right")
        eligible = block["views"]["all_error"][1]
        phases.append(values[eligible])
    return np.concatenate(phases)


def partition_rankings(labels, left, right, phases):
    """Every positive subset is compared against the same full negative set."""
    selections = {"all": np.ones(len(labels), dtype=bool)}
    for index, name in enumerate(PHASES):
        selections[name] = ~labels | (phases == index)
    selections["continuation"] = ~labels | (phases > 0)
    rankings = {}
    for name, selected in selections.items():
        rankings[name] = (
            selected,
            Ranking(labels[selected], left[selected]),
            Ranking(labels[selected], right[selected]),
        )
    return rankings


def measure_partitions(rankings, weights, positive_total):
    result = {}
    for name, (selected, left, right) in rankings.items():
        local = weights[selected]
        positives = float(local @ left.labels)
        negative_count = float(local.sum() - positives)
        first = left.measure(local)["auroc"]
        second = right.measure(local)["auroc"]
        difference = first - second if first is not None else None
        fraction = positives / positive_total if positive_total else None
        result[name] = {
            "positives": int(positives),
            "negatives": int(negative_count),
            "positive_weight": fraction,
            "left_auroc": first,
            "right_auroc": second,
            "delta_auroc": difference,
            "weighted_delta_auroc": fraction * difference
            if difference is not None
            else None,
        }
    return result


def bootstrap_partitions(rankings, labels, sources, draws, seed):
    """Resample complete sources; recompute positive weights in every draw."""
    unique, inverse = np.unique(sources, return_inverse=True)
    samples = {name: [] for name in rankings}
    random = np.random.default_rng(seed)
    for _ in range(draws if len(unique) > 1 else 0):
        counts = np.bincount(
            random.integers(len(unique), size=len(unique)), minlength=len(unique)
        )
        weights = counts[inverse]
        measured = measure_partitions(rankings, weights, float(weights @ labels))
        if measured["all"]["delta_auroc"] is None:
            continue
        for name, row in measured.items():
            # An absent positive phase has zero mass in a valid overall draw.
            # Its AUROC remains undefined; its contribution is exactly zero.
            contribution = row["weighted_delta_auroc"] if row["positives"] else 0.0
            samples[name].append(contribution)
    return {
        name: {
            "valid": len(values),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
        }
        for name, values in samples.items()
    }


def decompose_difference(blocks, left, right, draws, seed=17):
    """Partition a frozen-score AUROC difference on pairwise common coverage.

    Gold spans must already merge overlap while retaining adjacent annotations.
    Labels are used only to account for evaluated gain. AP is not additive.
    """
    labels, first, sources, _ = metric_arrays(blocks, "all_error", left)
    _, second, _, _ = metric_arrays(blocks, "all_error", right)
    covered = np.isfinite(first) & np.isfinite(second)
    phases = token_phases(blocks)[covered]
    labels, sources = labels[covered], sources[covered]
    rankings = partition_rankings(labels, first[covered], second[covered], phases)
    measured = measure_partitions(rankings, np.ones(len(labels)), float(labels.sum()))
    intervals = bootstrap_partitions(rankings, labels, sources, draws, seed)
    for name, row in measured.items():
        row["weighted_delta_source_bootstrap"] = intervals[name]
    total = measured["all"]["delta_auroc"]
    terms = [measured[name]["weighted_delta_auroc"] for name in PHASES]
    summed = (
        sum(value for value in terms if value is not None)
        if total is not None
        else None
    )
    return {
        "left": left,
        "right": right,
        "tokens": len(labels),
        "sources": len(np.unique(sources)),
        "partitions": measured,
        "component_order": list(PHASES),
        "weighted_delta_sum": summed,
        "residual": total - summed if total is not None else None,
        "interpretation": "Exact ranking decomposition with common negatives; not a causal effect fraction.",
        "missing_class": "Undefined AUROC remains null; absent-phase bootstrap contributions are zero when both overall classes exist.",
    }
