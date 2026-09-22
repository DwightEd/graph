"""Short-span evaluation of frozen scores; labels never change the scores."""

from collections import defaultdict
from itertools import combinations

import numpy as np

from experiments.unsupervised_token_graph.evaluate import Ranking
from experiments.unsupervised_token_graph.head_geometry.continuity_metrics import (
    _span_measurement,
    merge_token_spans,
    summarize_spans,
)

LENGTH_BINS = {"1-2": (1, 2), "3-4": (3, 4), "5-8": (5, 8),
               "1-8": (1, 8), "9+": (9, None)}
HISTORY_STEPS = 15


def normal_fpr_threshold(scores, budget):
    """Largest attainable normal alarm count within budget, with score >= threshold.

    Ties stay together. This uses test labels and is an operating-curve point,
    not a threshold calibrated for deployment or a guarantee on future FPR.
    """
    scores = np.asarray(scores, float)
    if not len(scores) or budget == 0:
        return float("inf")
    allowed = int(np.floor(budget * len(scores)))
    if allowed >= len(scores):
        return float("-inf")
    boundary = np.partition(scores, len(scores) - allowed - 1)[len(scores) - allowed - 1]
    return float(np.nextafter(boundary, np.inf))


def _common_blocks(blocks, methods):
    """All methods see identical measured positions; original blocks stay untouched."""
    result = []
    for block in blocks:
        finite = block["common_finite"].copy()
        for method in methods:
            finite &= np.isfinite(block["scores"][method])
        scores = {method: np.where(finite, block["scores"][method], np.nan)
                  for method in methods}
        result.append(dict(block, scores=scores, common_finite=finite))
    return result


def _normal_scores(blocks, method):
    values = []
    for block in blocks:
        labels, eligible = block["views"]["all_error"]
        selected = ~labels & eligible & block["common_finite"]
        values.append(block["scores"][method][selected])
    return np.concatenate(values)


def _alarm_blocks(blocks, method, threshold):
    if threshold is None:
        return blocks
    return [dict(block, alarms={method: block["scores"][method].astype(np.float64) >= threshold})
            for block in blocks]


def _normal_alarm_summary(blocks, method):
    observed = alarms = 0
    for block in blocks:
        labels, eligible = block["views"]["all_error"]
        selected = ~labels & eligible & block["common_finite"]
        observed += int(selected.sum())
        alarms += int(block["alarms"][method][selected].sum())
    return {"normal_tokens": observed, "normal_alarms": alarms,
            "achieved_normal_fpr": alarms / observed if observed else None}


def _measure_span(block, method, start, end):
    row = _span_measurement(block, method, start, end)
    scores = block["scores"][method][start:end]
    row["max_score"] = float(scores[np.isfinite(scores)].max()) if row["scored_tokens"] else None
    if row["observed_any_alarm"]:
        row["status"] = "detected"
    elif row["complete"]:
        row["status"] = "complete_miss"
    else:
        row["status"] = "unresolved"
    previous = block["views"]["all_error"][0][max(0, start - HISTORY_STEPS):start]
    row["recent_error_history"] = "recovery" if previous.any() else "clean_history"
    row["history_tokens"] = len(previous)
    return row


def _span_tables(blocks, method, metadata):
    spans, paired = [], []
    for block in blocks:
        identity = {"id": block["record"]["id"], "source_id": block["record"]["source_id"]}
        for index, (start, end) in enumerate(merge_token_spans(block["gold"])):
            spans.append(dict(metadata, **identity, span_index=index,
                              **_measure_span(block, method, start, end)))
        error_labels = block["views"]["all_error"][0]
        for pair in block["pairs"]:
            start = pair["normal_start"]
            previous_error = error_labels[max(0, start - HISTORY_STEPS):start].any()
            history = "recovery" if previous_error else "clean_history"
            for side in ("error", "normal"):
                row = _measure_span(block, method, pair[side + "_start"], pair[side + "_end"])
                paired.append(dict(pair, **metadata, **identity, side=side,
                                   normal_history=history, **row))
    return spans, paired


def _span_summary(rows):
    summary = summarize_spans(rows)
    count = len(rows)
    detected = summary["detected_spans"]
    unresolved = summary["incomplete_without_alarm"]
    complete = [row for row in rows if row["complete"]]
    summary.update(
        before_end_detected=detected,
        before_end_recall_lower_bound=detected / count if count else None,
        before_end_recall_upper_bound=(detected + unresolved) / count if count else None,
        complete_miss=summary["completely_observed_missed"], unresolved=unresolved,
        complete_span_recall=float(np.mean([row["observed_any_alarm"] for row in complete]))
        if complete else None,
        missing_tokens=summary["total_tokens"] - summary["scored_tokens"],
        onset_recall_lower_bound=summary["onset_alarms"] / count if count else None,
        span_weighting="one_per_span", delay_population="detected_only",
    )
    return summary


def _in_bin(row, limits):
    minimum, maximum = limits
    return row["length"] >= minimum and (maximum is None or row["length"] <= maximum)


def _paired_ranking(rows):
    """Pool complete paired span maxima; also report within-pair comparisons."""
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["id"], row["pair_id"])][row["side"]] = row
    complete = [pair for pair in pairs.values()
                if pair["error"]["complete"] and pair["normal"]["complete"]]
    error = np.asarray([pair["error"]["max_score"] for pair in complete], float)
    normal = np.asarray([pair["normal"]["max_score"] for pair in complete], float)
    labels = np.r_[np.ones(len(error), bool), np.zeros(len(normal), bool)]
    ranking = Ranking(labels, np.r_[error, normal]).measure()
    wins = (error > normal) + .5 * (error == normal)
    return {"pairs": len(pairs), "complete_pairs": len(complete),
            "incomplete_pairs": len(pairs) - len(complete),
            "matched_span_max_auroc": ranking["auroc"],
            "within_pair_win_rate": float(wins.mean()) if len(wins) else None,
            "ranking_population": "complete_pairs_only",
            "auroc_comparisons": "all_error_normal_span_pairs_in_matched_set"}


def _paired_summary(rows):
    result = {}
    for history in ("all", "clean_history", "recovery"):
        selected = rows if history == "all" else [row for row in rows if row["normal_history"] == history]
        result[history] = {"ranking": _paired_ranking(selected)}
        for side in ("error", "normal"):
            result[history][side] = _span_summary([row for row in selected if row["side"] == side])
    return result


def _setting(blocks, method, group, threshold, budget, setting):
    selected = _alarm_blocks(blocks, method, threshold)
    metadata = dict(group=group, method=method, setting=setting, requested_normal_fpr=budget)
    spans, paired = _span_tables(selected, method, metadata)
    result = dict(metadata, threshold=threshold, **_normal_alarm_summary(selected, method))
    result["threshold_source"] = "saved_frozen_alarms" if threshold is None else "test_normal_labels_descriptive_only"
    result["threshold_available"] = threshold is None or result["normal_tokens"] > 0
    result["bins"] = {}
    for name, limits in LENGTH_BINS.items():
        error_rows = [row for row in spans if _in_bin(row, limits)]
        pair_rows = [row for row in paired if _in_bin(row, limits)]
        history = {state: _span_summary([row for row in error_rows if row["recent_error_history"] == state])
                   for state in ("clean_history", "recovery")}
        result["bins"][name] = {"error": _span_summary(error_rows), "error_history": history,
                                "matched": _paired_summary(pair_rows)}
    return result, spans, paired


def _bootstrap_delta(left, right, metric, draws, random):
    """Equal-span paired difference, resampling entire sources together."""
    if metric == "onset_recall":
        eligible = [index for index, row in enumerate(left) if row["onset_observed"]]
        field = "onset_alarm"
    else:
        eligible = [index for index, row in enumerate(left)
                    if metric != "complete_span_recall" or row["complete"]]
        field = "observed_any_alarm"
    sources, inverse = np.unique([left[index]["source_id"] for index in eligible], return_inverse=True)
    difference = np.asarray([int(left[index][field]) - int(right[index][field]) for index in eligible])
    totals = np.bincount(inverse, weights=difference, minlength=len(sources))
    counts = np.bincount(inverse, minlength=len(sources))
    samples = []
    if len(sources) > 1 and draws:
        weights = random.multinomial(len(sources), np.full(len(sources), 1 / len(sources)), size=draws)
        samples = (weights @ totals) / (weights @ counts)
    return {"metric": metric, "delta": float(difference.mean()) if len(difference) else None,
            "spans": len(eligible), "sources": len(sources), "bootstrap_valid": len(samples),
            "ci95": np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
            "bootstrap_unit": "source", "estimand": "equal_span_paired_method_difference",
            "threshold_treatment": "fixed_at_reported_operating_point"}


def _comparisons(rows, methods, draws, seed):
    by_setting = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["length"] <= 8:
            key = row["setting"], row["requested_normal_fpr"]
            by_setting[key][row["method"]].append(row)
    result = []
    random = np.random.default_rng(seed)
    for (setting, budget), by_method in by_setting.items():
        for left, right in combinations(methods, 2):
            deltas = [_bootstrap_delta(by_method[left], by_method[right], metric, draws, random)
                      for metric in ("before_end_recall_lower_bound", "onset_recall", "complete_span_recall")]
            result.append(dict(setting=setting, requested_normal_fpr=budget, length_bin="1-8",
                               left=left, right=right, differences=deltas))
    return result


def _evaluate_group(blocks, methods, group, fpr_budgets, bootstrap, seed):
    results, spans, paired = {}, [], []
    for method in methods:
        frozen, error_rows, pair_rows = _setting(blocks, method, group, None, None, "frozen")
        spans.extend(error_rows)
        paired.extend(pair_rows)
        curve = []
        normal_scores = _normal_scores(blocks, method)
        for budget in fpr_budgets:
            threshold = normal_fpr_threshold(normal_scores, budget)
            measurement, error_rows, pair_rows = _setting(
                blocks, method, group, threshold, budget, "descriptive_test_normal_fpr_curve")
            curve.append(measurement)
            spans.extend(error_rows)
            paired.extend(pair_rows)
        results[method] = {"frozen": frozen, "descriptive_test_normal_fpr_curve": curve}
    comparisons = _comparisons(spans, methods, bootstrap, seed)
    record = blocks[0]["record"]
    identity = {key: record[key] for key in ("dataset", "task", "generator")}
    return dict(identity=identity, answers=len(blocks), methods=results, comparisons=comparisons), spans, paired


def evaluate(blocks, methods, fpr_budgets=(.01, .03, .05), bootstrap=200, seed=17):
    """Evaluate prepared continuity blocks; no classifier or score fitting.

    Ranges are half-open token intervals. All comparison rows use the selected
    methods' common finite coverage. Preserve original coverage in the caller.
    """
    grouped = defaultdict(list)
    for block in _common_blocks(blocks, methods):
        record = block["record"]
        key = "|".join(str(record[field]) for field in ("dataset", "task", "generator"))
        grouped[key].append(block)
    report = {"groups": {}, "span_rows": [], "paired_rows": []}
    for group, selected in grouped.items():
        result, spans, paired = _evaluate_group(selected, methods, group, fpr_budgets, bootstrap, seed)
        report["groups"][group] = result
        report["span_rows"].extend(spans)
        report["paired_rows"].extend(paired)
    return report
