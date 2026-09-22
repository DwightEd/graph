"""Equal-span and equal-pair method differences, resampling whole sources."""

from collections import defaultdict
from itertools import combinations

import numpy as np


def complete_pairs(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["id"], row["pair_id"])][row["side"]] = row
    complete = {
        key: pair
        for key, pair in pairs.items()
        if pair["error"]["complete"] and pair["normal"]["complete"]
    }
    return pairs, complete


def _bootstrap_values(difference, source_ids, metric, draws, random):
    sources, inverse = np.unique(source_ids, return_inverse=True)
    difference = np.asarray(difference)
    totals = np.bincount(inverse, weights=difference, minlength=len(sources))
    counts = np.bincount(inverse, minlength=len(sources))
    samples = []
    if len(sources) > 1 and draws:
        weights = random.multinomial(
            len(sources), np.full(len(sources), 1 / len(sources)), size=draws
        )
        samples = (weights @ totals) / (weights @ counts)
    return {
        "metric": metric,
        "delta": float(difference.mean()) if len(difference) else None,
        "spans": len(difference),
        "sources": len(sources),
        "bootstrap_valid": len(samples),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist() if len(samples) else None,
        "bootstrap_unit": "source",
        "estimand": "equal_span_paired_method_difference",
        "threshold_treatment": "fixed_at_reported_operating_point",
    }


def _bootstrap_delta(left, right, metric, draws, random):
    """Equal-span paired difference, resampling entire sources together."""
    if metric == "onset_recall":
        eligible = [index for index, row in enumerate(left) if row["onset_observed"]]
        field = "onset_alarm"
    else:
        eligible = [
            index
            for index, row in enumerate(left)
            if metric != "complete_span_recall" or row["complete"]
        ]
        field = "observed_any_alarm"
    differences = [int(left[index][field]) - int(right[index][field]) for index in eligible]
    sources = [left[index]["source_id"] for index in eligible]
    return _bootstrap_values(differences, sources, metric, draws, random)


def _matched_deltas(left_pairs, right_pairs, draws, random):
    result = []
    keys = sorted(left_pairs.keys() & right_pairs.keys())
    sources = [left_pairs[key]["error"]["source_id"] for key in keys]
    error = [
        int(left_pairs[key]["error"]["observed_any_alarm"])
        - int(right_pairs[key]["error"]["observed_any_alarm"])
        for key in keys
    ]
    normal = [
        int(left_pairs[key]["normal"]["observed_any_alarm"])
        - int(right_pairs[key]["normal"]["observed_any_alarm"])
        for key in keys
    ]
    advantage = np.asarray(error) - np.asarray(normal)
    for metric, differences in (
        ("matched_error_recall", error),
        ("matched_normal_fpr", normal),
        ("matched_recall_minus_fpr", advantage),
    ):
        measured = _bootstrap_values(differences, sources, metric, draws, random)
        measured["pairs"] = measured.pop("spans")
        measured["estimand"] = "equal_complete_pair_method_difference"
        result.append(measured)
    return result


def compare_matched_methods(rows, methods, draws, seed):
    """Compare false alarms alongside recall on identical complete matched pairs."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["length"] <= 8:
            grouped[row["setting"], row["requested_normal_fpr"]][row["method"]].append(row)
    result = []
    random = np.random.default_rng(seed)
    for (setting, budget), by_method in grouped.items():
        for left, right in combinations(methods, 2):
            left_pairs = complete_pairs(by_method[left])[1]
            right_pairs = complete_pairs(by_method[right])[1]
            deltas = _matched_deltas(left_pairs, right_pairs, draws, random)
            for measured in deltas:
                result.append(
                    {
                        "setting": setting,
                        "requested_normal_fpr": budget,
                        "length_bin": "1-8",
                        "left": left,
                        "right": right,
                        "differences": [measured],
                    }
                )
    return result


def compare_methods(rows, methods, draws, seed):
    by_setting = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["length"] <= 8:
            key = row["setting"], row["requested_normal_fpr"]
            by_setting[key][row["method"]].append(row)
    result = []
    random = np.random.default_rng(seed)
    for (setting, budget), by_method in by_setting.items():
        for left, right in combinations(methods, 2):
            deltas = [
                _bootstrap_delta(by_method[left], by_method[right], metric, draws, random)
                for metric in (
                    "before_end_recall_lower_bound",
                    "onset_recall",
                    "complete_span_recall",
                )
            ]
            result.append(
                {
                    "setting": setting,
                    "requested_normal_fpr": budget,
                    "length_bin": "1-8",
                    "left": left,
                    "right": right,
                    "differences": deltas,
                }
            )
    return result
