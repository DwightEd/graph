"""Ranking, localization and false alarms against official token annotations."""

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.native_support.evaluate import ranking


def grouped_indices(groups):
    order = np.argsort(groups, kind="stable")
    cuts = np.flatnonzero(np.diff(groups[order])) + 1
    return np.split(order, cuts)


def pairwise_within(labels, scores, groups):
    total, correct = 0, 0.
    for selected in grouped_indices(groups):
        target = labels[selected]
        pairs = int(target.sum()) * int(len(target) - target.sum())
        if pairs:
            correct += pairs * roc_auc_score(target, scores[selected])
            total += pairs
    return float(correct / total) if total else None


def threshold_at_fpr(labels, scores, fpr=.05):
    if not (labels == 0).any():
        return None
    return float(np.quantile(scores[labels == 0], 1 - fpr, method="higher"))


def evaluate_method(pack, scores, threshold):
    target = pack["labels"]
    result = ranking(target, scores)
    result["within_answer_auroc"] = pairwise_within(target, scores, pack["answer_index"])
    result["within_unit_auroc"] = pairwise_within(target, scores,
        pack["answer_index"].astype(np.int64) * (int(pack["unit_index"].max()) + 1) + pack["unit_index"])
    alarm = scores > threshold if threshold is not None else None
    result.update(threshold=threshold, threshold_rule="dev_pooled_normal_token_95th_percentile_strict_gt",
        token_fpr=float(alarm[target == 0].mean()) if alarm is not None and (target == 0).any() else None,
        token_recall=float(alarm[target == 1].mean()) if alarm is not None and target.any() else None)
    for key in ("onsets", "firsts"):
        mask = (target == 0) | pack[key]
        result[key] = ranking(pack[key][mask].astype(int), scores[mask])
        result[key]["recall"] = float(alarm[pack[key]].mean()) if alarm is not None and pack[key].any() else None
    normal_answers = []
    for selected in grouped_indices(pack["answer_index"]):
        if alarm is not None and not target[selected].any():
            normal_answers.append(bool(alarm[selected].any()))
    result["normal_answers"] = len(normal_answers)
    result["normal_answer_false_alarm"] = float(np.mean(normal_answers)) if normal_answers else None
    return result


def subset(pack, selected):
    return {name: value[selected] for name, value in pack.items()}


def evaluate_all(pack, scores, thresholds=None, detailed=True):
    result = {}
    for method, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{method}: scores are incomplete or nonfinite")
        threshold = threshold_at_fpr(pack["labels"], values) if thresholds is None else thresholds[method]
        if detailed:
            result[method] = evaluate_method(pack, values, threshold)
        else:
            result[method] = dict(ranking(pack["labels"], values), threshold=threshold)
    return result


def source_bootstrap(pack, candidate, baseline, repeats=300, seed=42):
    """Resample complete source clusters; return AUROC difference uncertainty."""
    identities, inverse = np.unique(pack["source_index"], return_inverse=True)
    rng = np.random.default_rng(seed)
    differences = []
    prepared = [weighted_auc_order(pack["labels"], scores) for scores in (candidate, baseline)]
    for _ in range(repeats):
        count = np.bincount(rng.integers(0, len(identities), len(identities)), minlength=len(identities))
        weight = count[inverse]
        if not np.unique(pack["labels"][weight > 0]).size == 2:
            continue
        difference = weighted_auc(prepared[0], weight) - weighted_auc(prepared[1], weight)
        differences.append(difference)
    interval = np.quantile(differences, [.025, .975]).tolist() if differences else None
    return dict(repeats=repeats, valid_repeats=len(differences), seed=seed,
                auroc_delta_95ci=interval, cluster="source", exploratory=True)


def weighted_auc_order(labels, scores):
    """Cache ordering and exact tie groups once for repeated cluster weighting."""
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    return order, starts, labels[order]


def weighted_auc(prepared, weights):
    order, starts, labels = prepared
    ordered_weight = weights[order]
    positive = np.add.reduceat(ordered_weight * labels, starts)
    negative = np.add.reduceat(ordered_weight * (1 - labels), starts)
    negatives_before = np.cumsum(negative) - negative
    return float(np.sum(positive * (negatives_before + .5 * negative)) / (positive.sum() * negative.sum()))
