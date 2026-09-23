"""Exact tied-score AP accounting and fixed-budget ranking audits; no fitting."""

from itertools import pairwise

import numpy as np

from .comparison_evaluation import group_metrics, phase_masks
from .evaluate import ranking


def rank_ledger(labels, scores):
    """AP uses precision AFTER each complete tie group, matching sklearn.

    Each positive receives precision_at_its_threshold / total_positives.
    Summing these credits over any partition exactly recovers pooled AP.
    """
    values, inverse, sizes = np.unique(scores, return_inverse=True, return_counts=True)
    positive = np.bincount(inverse, weights=labels, minlength=len(values)).astype(int)
    total_above = np.cumsum(sizes[::-1])[::-1] - sizes
    positive_above = np.cumsum(positive[::-1])[::-1] - positive
    precision = (positive_above + positive) / (total_above + sizes)
    return {
        "rank_first": (total_above + 1)[inverse],
        "rank_last": (total_above + sizes)[inverse],
        "tie_tokens": sizes[inverse], "tie_errors": positive[inverse],
        "normals_above": (total_above - positive_above)[inverse],
        "precision_at_threshold": precision[inverse],
        "ap_credit": labels * precision[inverse] / max(int(labels.sum()), 1),
    }


def scopes(tokens):
    labels = np.asarray([row["label"] for row in tokens])
    onset = np.asarray([row["is_span_onset"] for row in tokens], dtype=bool)
    first = np.asarray([row["is_answer_first_error"] for row in tokens], dtype=bool)
    result = phase_masks(labels, onset, first)
    front = np.asarray([row["half"] == "front" for row in tokens])
    probability = np.asarray([row["state_dynamics"] for row in tokens])
    result.update(front_half=(front, labels), back_half=(~front, labels),
                  E_mode=(probability < .5, labels), H_mode=(probability >= .5, labels),
                  H_ge_0_9=(probability >= .9, labels), H_exact_1=(probability == 1, labels))
    return result


def metric_rows(tokens, methods):
    answers = np.asarray([row["response_id"] for row in tokens])
    sources = np.asarray([row["source_id"] for row in tokens])
    result = []
    for method in methods:
        score = np.asarray([row[method] for row in tokens])
        for scope, (mask, labels) in scopes(tokens).items():
            metrics = group_metrics(labels[mask], score[mask], answers[mask], sources[mask])
            result.append({"method": method, "scope": scope,
                           **{key: metrics[key] for key in ("tokens", "positives", "negatives", "auroc", "ap")},
                           "source_auroc": metrics["source_balanced"]["auroc"],
                           "source_ap": metrics["source_balanced"]["ap"],
                           "within_answer_auroc": metrics["within_answer"]["pair_weighted_auroc"]})
    return result


def budget_rows(labels, scores, method):
    """Expected top-k under random ordering INSIDE boundary ties; bounds included."""
    count = len(labels)
    budgets = {min(count, value) for value in (50, 100)}
    budgets.update(max(1, int(np.ceil(count * fraction))) for fraction in (.01, .05, .1, .2))
    result = []
    for budget in sorted(budgets):
        threshold = np.sort(scores)[-budget]
        above, tied = scores > threshold, scores == threshold
        slots = budget - int(above.sum())
        tied_errors = int(labels[tied].sum())
        found = int(labels[above].sum())
        expected = found + slots * tied_errors / int(tied.sum())
        result.append({"method": method, "budget": budget, "threshold": float(threshold),
                       "strictly_above": int(above.sum()), "boundary_ties": int(tied.sum()),
                       "boundary_errors": tied_errors, "boundary_slots": slots,
                       "expected_errors": expected, "expected_normals": budget - expected,
                       "precision_expected": expected / budget,
                       "precision_min": (found + max(0, slots - int(tied.sum()) + tied_errors)) / budget,
                       "precision_max": (found + min(slots, tied_errors)) / budget,
                       "recall_expected": expected / int(labels.sum()) if labels.sum() else None})
    return result


def threshold_rows(labels, scores, method):
    values, inverse, sizes = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse, weights=labels, minlength=len(values)).astype(int)
    total = np.cumsum(sizes[::-1])[::-1]
    found = np.cumsum(positives[::-1])[::-1]
    result = []
    for index in range(len(values) - 1, -1, -1):
        recall = found[index] / int(labels.sum()) if labels.sum() else None
        result.append({"method": method, "threshold": float(values[index]),
                       "selected": int(total[index]), "errors": int(found[index]),
                       "normals": int(total[index] - found[index]),
                       "precision": float(found[index] / total[index]), "recall": recall,
                       "tie_tokens": int(sizes[index]), "tie_errors": int(positives[index])})
    return result


def ap_parts(tokens, credit, method):
    """These are contributions to GLOBAL AP, not AP recomputed in a subgroup."""
    partitions = {"phase": np.asarray([row["phase"] for row in tokens]),
                  "half": np.asarray([row["half"] for row in tokens]),
                  "answer": np.asarray([row["response_id"] for row in tokens])}
    labels = np.asarray([row["label"] for row in tokens])
    result = []
    for partition, groups in partitions.items():
        for group in np.unique(groups):
            selected = groups == group
            result.append({"method": method, "partition": partition, "group": str(group),
                           "tokens": int(selected.sum()), "errors": int(labels[selected].sum()),
                           "global_ap_credit": float(credit[selected].sum())})
    return result


def half_pair_rows(tokens, scores, method):
    """Decompose pooled AUC into positive-half x negative-half pairs."""
    labels = np.asarray([row["label"] for row in tokens])
    halves = np.asarray([row["half"] for row in tokens])
    total_pairs = int(labels.sum()) * int((labels == 0).sum())
    result = []
    for positive_half in ("front", "back"):
        positives = scores[(labels == 1) & (halves == positive_half)]
        for negative_half in ("front", "back"):
            negatives = np.sort(scores[(labels == 0) & (halves == negative_half)])
            lower = np.searchsorted(negatives, positives, side="left")
            upper = np.searchsorted(negatives, positives, side="right")
            wins = float((.5 * (lower + upper)).sum())
            pairs = len(positives) * len(negatives)
            result.append({"method": method, "positive_half": positive_half, "negative_half": negative_half,
                           "pairs": pairs, "auroc": wins / pairs if pairs else None,
                           "global_auc_credit": wins / total_pairs if total_pairs else None})
    return result


def ap_delta_rows(parts, controls):
    lookup = {(row["method"], row["partition"], row["group"]): row for row in parts}
    result = []
    for row in parts:
        if row["method"] != "state_dynamics":
            continue
        for control in controls:
            baseline = lookup[control, row["partition"], row["group"]]
            result.append({"candidate": "state_dynamics", "control": control,
                           "partition": row["partition"], "group": row["group"], "errors": row["errors"],
                           "candidate_ap_credit": row["global_ap_credit"],
                           "control_ap_credit": baseline["global_ap_credit"],
                           "delta_global_ap": row["global_ap_credit"] - baseline["global_ap_credit"]})
    return result


def mode_bin_rows(tokens):
    probability = np.asarray([row["state_dynamics"] for row in tokens])
    labels = np.asarray([row["label"] for row in tokens])
    edges = (0., .1, .5, .9, 1.)
    result = []
    for left, right in pairwise(edges):
        selected = (probability >= left) & (probability < right)
        result.append({"lower": left, "upper_exclusive": right, "exact_one": False,
                       "tokens": int(selected.sum()), "errors": int(labels[selected].sum()),
                       "normals": int((1 - labels[selected]).sum()),
                       "error_fraction": float(labels[selected].mean()) if selected.any() else None})
    selected = probability == 1
    result.append({"lower": 1., "upper_exclusive": None, "exact_one": True,
                   "tokens": int(selected.sum()), "errors": int(labels[selected].sum()),
                   "normals": int((1 - labels[selected]).sum()),
                   "error_fraction": float(labels[selected].mean()) if selected.any() else None})
    return result


def ranking_tables(tokens, methods):
    labels = np.asarray([row["label"] for row in tokens])
    result = {name: [] for name in ("ranking", "top_budget", "precision_recall", "ap_parts", "auc_pairs", "score_distribution")}
    identity = ("response_id", "source_id", "target", "token", "label", "phase", "half")
    for method in methods:
        scores = np.asarray([row[method] for row in tokens])
        ledger = rank_ledger(labels, scores)
        result["ranking"].extend({**{name: row[name] for name in identity}, "method": method,
                                  "score": float(scores[index]),
                                  **{name: value[index].item() for name, value in ledger.items()}}
                                 for index, row in enumerate(tokens))
        result["top_budget"].extend(budget_rows(labels, scores, method))
        result["precision_recall"].extend(threshold_rows(labels, scores, method))
        result["ap_parts"].extend(ap_parts(tokens, ledger["ap_credit"], method))
        result["auc_pairs"].extend(half_pair_rows(tokens, scores, method))
        for label in (0, 1):
            values = scores[labels == label]
            unique, sizes = np.unique(values, return_counts=True)
            result["score_distribution"].append({"method": method, "label": label, "tokens": len(values),
                "unique": len(unique), "tied_tokens": int(sizes[sizes > 1].sum()),
                "exact_zero": int((values == 0).sum()), "exact_one": int((values == 1).sum()),
                **{f"q{int(q * 100):02d}": float(np.quantile(values, q)) if len(values) else None
                   for q in (0, .1, .5, .9, .99, 1)}})
        measured = ranking(labels, scores)["ap"]
        if measured is not None and not np.isclose(ledger["ap_credit"].sum(), measured, atol=1e-12, rtol=0):
            raise AssertionError(f"{method}: AP accounting does not match evaluation")
    result["metrics"] = metric_rows(tokens, methods)
    result["ap_deltas"] = ap_delta_rows(result["ap_parts"],
                                       [name for name in methods if name != "state_dynamics"])
    result["mode_bins"] = mode_bin_rows(tokens)
    return result
