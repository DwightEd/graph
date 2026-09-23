"""Paired whole-source bootstrap; fixed scores and no method selection."""

import numpy as np
from tqdm import trange

from .transport_audit_rank import CANDIDATE, CONTROLS, audit_scopes


def weighted_ranks(labels, score_bins, weights):
    """Weighted AUROC/AP using complete tie groups, including absent classes."""
    positive = np.bincount(score_bins, weights=weights * labels, minlength=score_bins.max() + 1)
    negative = np.bincount(score_bins, weights=weights * (1 - labels), minlength=len(positive))
    positives, negatives = positive.sum(), negative.sum()
    auc = np.nan
    ap = np.nan
    if positives and negatives:
        auc = np.sum(positive * (np.cumsum(negative) - negative / 2)) / (positives * negatives)
    if positives:
        found = np.cumsum(positive[::-1])
        total = found + np.cumsum(negative[::-1])
        precision = np.divide(found, total, out=np.zeros_like(found), where=total > 0)
        ap = np.sum(positive[::-1] * precision) / positives
    return np.asarray([auc, ap])


def prepare_scopes(tokens, methods):
    _, sources = np.unique([row["source_id"] for row in tokens], return_inverse=True)
    result = {}
    for scope, (mask, labels) in audit_scopes(tokens).items():
        if not mask.any():
            continue
        bins = {}
        for method in methods:
            scores = np.asarray([row[method] for row in tokens])[mask]
            bins[method] = np.unique(scores, return_inverse=True)[1]
        result[scope] = (labels[mask].astype(int), sources[mask], bins)
    return result, len(np.unique(sources))


def paired_bootstrap(tokens, replicates=1000, seed=37):
    methods = (CANDIDATE, *CONTROLS)
    prepared, source_count = prepare_scopes(tokens, methods)
    scope_names = list(prepared)
    deltas = np.full((replicates, len(scope_names), len(CONTROLS), 2), np.nan)
    rng = np.random.default_rng(seed)
    for draw in trange(replicates, desc="paired source bootstrap", disable=replicates == 0):
        multiplicity = np.bincount(rng.integers(source_count, size=source_count), minlength=source_count)
        for index, (labels, sources, bins) in enumerate(prepared.values()):
            metrics = {method: weighted_ranks(labels, bins[method], multiplicity[sources]) for method in methods}
            for column, control in enumerate(CONTROLS):
                deltas[draw, index, column] = metrics[CANDIDATE] - metrics[control]
    rows = interval_rows(prepared, deltas, replicates, source_count)
    arrays = {"deltas": deltas, "scopes": np.asarray(scope_names), "controls": np.asarray(CONTROLS),
              "metrics": np.asarray(["auroc", "ap"]), "seed": np.asarray(seed)}
    return rows, arrays


def interval_rows(prepared, deltas, replicates, source_count):
    rows = []
    for index, (scope, (labels, sources, bins)) in enumerate(prepared.items()):
        point = {method: weighted_ranks(labels, value, np.ones(len(labels))) for method, value in bins.items()}
        for column, control in enumerate(CONTROLS):
            for metric, name in enumerate(("auroc", "ap")):
                values = deltas[:, index, column, metric]
                values = values[np.isfinite(values)]
                estimate = point[CANDIDATE][metric] - point[control][metric]
                enough = source_count > 1 and len(values) > 1
                interval = np.quantile(values, [.025, .975]) if enough else (None, None)
                rows.append({"scope": scope, "control": control, "metric": name,
                    "delta": float(estimate) if np.isfinite(estimate) else None,
                    "ci_low": interval[0], "ci_high": interval[1],
                    "replicates": replicates, "valid_replicates": len(values),
                    "source_clusters": source_count, "scope_sources": len(np.unique(sources)),
                    "status": "available" if enough else "insufficient_clusters_or_draws"})
    return rows
