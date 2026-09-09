"""Source-balanced two-stage estimates for lookback incidence and transport."""

from collections import defaultdict

import numpy as np


LABELS = (("N", 0), ("H", 1))


def _estimate(values, bootstrap, seed):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "ci95": [float("nan"), float("nan")],
        "sources": int(len(finite)),
    }
    if len(finite) > 1 and bootstrap:
        rng = np.random.default_rng(seed)
        draws = finite[rng.integers(0, len(finite), size=(bootstrap, len(finite)))].mean(1)
        result["ci95"] = np.quantile(draws, (0.025, 0.975)).tolist()
    return result


def _difference(first, second, bootstrap, seed):
    sources = sorted(set(first) | set(second))
    values = np.asarray(
        [[first.get(source, np.nan), second.get(source, np.nan)] for source in sources],
        dtype=float,
    )
    means = np.full(2, np.nan)
    for column in range(2):
        finite = values[:, column][np.isfinite(values[:, column])]
        if len(finite):
            means[column] = finite.mean()
    result = {
        "mean": float(means[0] - means[1]),
        "ci95": [float("nan"), float("nan")],
        "sources_H": int(np.isfinite(values[:, 0]).sum()) if len(values) else 0,
        "sources_N": int(np.isfinite(values[:, 1]).sum()) if len(values) else 0,
    }
    if len(values) > 1 and bootstrap:
        rng = np.random.default_rng(seed)
        draws = []
        for take in rng.integers(0, len(values), size=(bootstrap, len(values))):
            sample = values[take]
            average = np.nanmean(sample, axis=0)
            if np.isfinite(average).all():
                draws.append(average[0] - average[1])
        if draws:
            result["ci95"] = np.quantile(draws, (0.025, 0.975)).tolist()
    return result


def summarize_hurdle(samples, *, bootstrap=200):
    """Estimate event incidence separately from transport given an event.

    Each record supplies one sample's source ID, labels, ordinary-token mask,
    prior-event indicator, and a label-free transport score. Source-level
    numerators and denominators are pooled before sources receive equal weight.
    Missing transport is never imputed as a zero response.
    """

    if bootstrap < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    counts = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=int)))
    transport = defaultdict(list)
    for sample in samples:
        labels = np.asarray(sample["labels"])
        ordinary = np.asarray(sample["ordinary"], dtype=bool)
        event = np.asarray(sample["event"], dtype=bool)
        score = np.asarray(sample["transport"], dtype=float)
        if not (labels.shape == ordinary.shape == event.shape == score.shape):
            raise ValueError("labels, ordinary, event and transport must share one token axis")
        if np.any(np.isfinite(score) & ~event):
            raise ValueError("conditional transport cannot exist without a prior event")
        group, source = str(sample["group"]), str(sample["source"])
        if not group or not source:
            raise ValueError("group and source identities are required")
        for name, value in LABELS:
            selected = ordinary & (labels == value)
            scored = selected & np.isfinite(score)
            counts[group, name][source] += (
                int(selected.sum()),
                int((selected & event).sum()),
                int(scored.sum()),
            )
            if scored.any():
                transport[group, name, source].extend(score[scored].tolist())

    result = {}
    for group in sorted({key[0] for key in counts}):
        group_result = {
            "estimand": "event incidence plus transport conditional on a prior event"
        }
        incidence_by_label = {}
        transport_by_label = {}
        for index, (name, _) in enumerate(LABELS):
            by_source = counts[group, name]
            incidence = {
                source: values[1] / values[0]
                for source, values in by_source.items()
                if values[0]
            }
            conditional = {
                source: float(np.mean(transport[group, name, source]))
                for source in by_source
                if transport[group, name, source]
            }
            totals = sum(by_source.values(), np.zeros(3, dtype=int))
            events = int(totals[1])
            group_result[name] = {
                "tokens": int(totals[0]),
                "events": events,
                "transport_scored": int(totals[2]),
                "transport_coverage_given_event": (
                    float(totals[2] / events) if events else float("nan")
                ),
                "event_rate": _estimate(list(incidence.values()), bootstrap, 180 + index),
                "transport_given_event": _estimate(
                    list(conditional.values()), bootstrap, 280 + index
                ),
            }
            incidence_by_label[name] = incidence
            transport_by_label[name] = conditional
        group_result["H_minus_N"] = {
            "event_rate": _difference(
                incidence_by_label["H"], incidence_by_label["N"], bootstrap, 380
            ),
            "transport_given_event": _difference(
                transport_by_label["H"], transport_by_label["N"], bootstrap, 480
            ),
        }
        result[group] = group_result
    return result
