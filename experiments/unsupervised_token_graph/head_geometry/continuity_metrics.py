"""Label-assisted continuity diagnostics on already frozen token scores."""

from collections import defaultdict

import numpy as np

from ..evaluate import Ranking


def merge_token_spans(spans):
    """Keep adjacent annotations distinct; merge annotations sharing a token."""
    merged = []
    for start, end in sorted(map(tuple, spans)):
        if merged and start < merged[-1][1]:
            previous, previous_end = merged.pop()
            start, end = previous, max(previous_end, end)
        merged.append((int(start), int(end)))
    return np.asarray(merged, int).reshape(-1, 2)


def annotation_summary(blocks, method):
    counts = defaultdict(int)
    counts["fully_scored_spans"] = 0
    lengths = []
    for block in blocks:
        spans = merge_token_spans(block["gold"])
        error = np.zeros(block["tokens"], bool)
        onset = np.zeros(block["tokens"], bool)
        finite = np.isfinite(block["scores"][method])
        for start, end in spans:
            error[start:end], onset[start] = True, True
            lengths.append(int(end - start))
            counts["fully_scored_spans"] += int(finite[start:end].all())
        counts["input_token_spans"] += len(block["gold"])
        counts["token_spans"] += len(spans)
        counts["positive_runs"] += int((error & ~np.r_[False, error[:-1]]).sum())
        for name, mask in (("tokens", np.ones(len(error), bool)),
                           ("error_tokens", error), ("onsets", onset),
                           ("continuation_tokens", error & ~onset)):
            counts[name] += int(mask.sum())
            counts["scored_" + name] += int((mask & finite).sum())
    values, frequencies = np.unique(lengths, return_counts=True)
    result = dict(counts, answers=len(blocks))
    for prefix in ("", "scored_"):
        denominator = counts[prefix + "error_tokens"]
        numerator = counts[prefix + "continuation_tokens"]
        result[prefix + "continuation_fraction"] = numerator / denominator if denominator else None
        result[prefix + "onset_fraction"] = counts[prefix + "onsets"] / denominator if denominator else None
    result["length_histogram"] = dict(zip(map(str, values), map(int, frequencies)))
    result["length_quantiles"] = dict(zip(("min", "p25", "median", "p75", "p90", "max"),
        np.quantile(lengths, [0, .25, .5, .75, .9, 1]).tolist())) if lengths else None
    return result


def answer_rankings(blocks, method, view="all_error", common_with=None):
    """One row per answer, including answers excluded from within-answer ranking."""
    rows = []
    for block in blocks:
        labels, eligible = block["views"][view]
        scores = block["scores"][method]
        finite = eligible & np.isfinite(scores)
        if common_with is not None:
            finite &= np.isfinite(block["scores"][common_with])
        positives = int(labels[finite].sum())
        negatives = int(finite.sum()) - positives
        rows.append(dict(id=block["record"]["id"], source_id=block["record"]["source_id"],
            method=method, view=view, eligible_tokens=int(eligible.sum()),
            eligible_positives=int(labels[eligible].sum()), scored_tokens=int(finite.sum()),
            positives=positives, negatives=negatives, pairs=positives * negatives,
            **Ranking(labels[finite], scores[finite]).measure()))
    return rows


def _within_estimates(rows, weights):
    selected = np.asarray([row["pairs"] > 0 for row in rows])
    active = selected & (weights > 0)
    if not active.any():
        return {"pair_weighted_auroc": None, "macro_auroc": None, "macro_ap": None}
    indices = np.flatnonzero(active)
    answer_weights = weights[active]
    pair_weights = np.asarray([rows[index]["pairs"] for index in indices]) * answer_weights
    auc = np.asarray([rows[index]["auroc"] for index in indices])
    ap = np.asarray([rows[index]["ap"] for index in indices])
    return {"pair_weighted_auroc": float(np.average(auc, weights=pair_weights)),
                "macro_auroc": float(np.average(auc, weights=answer_weights)),
                "macro_ap": float(np.average(ap, weights=answer_weights))}


def within_answer_summary(rows):
    """Pair-weighted AUROC uses only positive-negative pairs from the same answer."""
    included = [row for row in rows if row["pairs"] > 0]
    excluded = [row for row in rows if row["pairs"] == 0]
    result = _within_estimates(rows, np.ones(len(rows)))
    result.update(answers=len(rows), included_answers=len(included), excluded_answers=len(excluded),
        eligible_tokens=sum(row["eligible_tokens"] for row in rows),
        scored_tokens=sum(row["scored_tokens"] for row in rows),
        included_scored_tokens=sum(row["scored_tokens"] for row in included),
        excluded_scored_tokens=sum(row["scored_tokens"] for row in excluded),
        within_answer_pairs=sum(row["pairs"] for row in included),
        no_score_answers=sum(row["scored_tokens"] == 0 for row in rows),
        scored_all_normal_answers=sum(row["positives"] == 0 and row["negatives"] > 0 for row in rows),
        scored_all_error_answers=sum(row["negatives"] == 0 and row["positives"] > 0 for row in rows))
    result["mixed_answers_lost_to_missing"] = sum(
        0 < row["eligible_positives"] < row["eligible_tokens"] and row["pairs"] == 0 for row in rows)
    return result


def within_answer_difference(blocks, view, left, right, draws):
    """Paired source bootstrap; both methods use the same tokens in each answer."""
    first = answer_rankings(blocks, left, view, common_with=right)
    second = answer_rankings(blocks, right, view, common_with=left)
    sources, inverse = np.unique([row["source_id"] for row in first], return_inverse=True)
    unit_weights = np.ones(len(first))
    a, b = _within_estimates(first, unit_weights), _within_estimates(second, unit_weights)
    order = list(a)
    delta = {key: a[key] - b[key] if a[key] is not None else None for key in order}
    random = np.random.default_rng(20260922)
    samples = []
    for _ in range(draws if len(sources) > 1 else 0):
        counts = np.bincount(random.integers(len(sources), size=len(sources)), minlength=len(sources))
        a = _within_estimates(first, counts[inverse])
        b = _within_estimates(second, counts[inverse])
        if a[order[0]] is not None:
            samples.append([a[key] - b[key] for key in order])
    return {"left": left, "right": right, "view": view, "delta": delta, "order": order,
                "sources": len(sources), "coverage": within_answer_summary(first),
                "bootstrap_unit": "source", "bootstrap_valid": len(samples),
                "ci95": np.quantile(samples, [.025, .975], axis=0).tolist() if samples else None}


def _span_measurement(block, method, start, end):
    scores = block["scores"][method][start:end]
    finite = np.isfinite(scores)
    alarms = block["alarms"][method][start:end] & finite
    detected = np.flatnonzero(alarms)
    delay = int(detected[0]) if len(detected) else None
    length, observed = len(scores), int(finite.sum())
    return {"start": int(start), "end": int(end), "length": length, "scored_tokens": observed,
        "missing_tokens": length - observed, "complete": bool(finite.all()),
        "mean_score": float(scores[finite].mean()) if observed else None,
        "onset_score": float(scores[0]) if finite[0] else None,
        "onset_observed": bool(finite[0]), "onset_alarm": bool(alarms[0]) if finite[0] else None,
        "alarms": int(alarms.sum()), "observed_any_alarm": bool(len(detected)),
        "token_alarm_rate": float(alarms.sum() / observed) if observed else None,
        "token_coverage_lower_bound": float(alarms.mean()),
        "first_observed_alarm_delay": delay,
        "missing_before_alarm": int((~finite[:delay]).sum()) if delay is not None else None}


def span_rows(blocks, method):
    rows = []
    for block in blocks:
        for index, (start, end) in enumerate(merge_token_spans(block["gold"])):
            rows.append(dict(id=block["record"]["id"], source_id=block["record"]["source_id"],
                method=method, span_index=index, **_span_measurement(block, method, start, end)))
    return rows


def summarize_spans(rows):
    onset = [row for row in rows if row["onset_observed"]]
    scored = [row for row in rows if row["scored_tokens"]]
    detected = [row for row in rows if row["observed_any_alarm"]]
    known_delay = [row for row in detected if row["missing_before_alarm"] == 0]
    complete = [row for row in rows if row["complete"]]
    alarms = sum(row["alarms"] for row in rows)
    scored_tokens = sum(row["scored_tokens"] for row in rows)
    total_tokens = sum(row["length"] for row in rows)
    return {"spans": len(rows), "complete_spans": len(complete), "scored_spans": len(scored),
        "no_score_spans": len(rows) - len(scored), "observed_onsets": len(onset),
        "missing_onsets": len(rows) - len(onset), "onset_alarms": sum(row["onset_alarm"] for row in onset),
        "onset_recall": sum(row["onset_alarm"] for row in onset) / len(onset) if onset else None,
        "detected_spans": len(detected), "any_alarm_rate": len(detected) / len(scored) if scored else None,
        "completely_observed_missed": sum(not row["observed_any_alarm"] for row in complete),
        "incomplete_without_alarm": sum(not row["complete"] and not row["observed_any_alarm"] for row in rows),
        "total_tokens": total_tokens, "scored_tokens": scored_tokens, "alarms": alarms,
        "token_alarm_rate": alarms / scored_tokens if scored_tokens else None,
        "mean_span_alarm_rate": float(np.mean([row["token_alarm_rate"] for row in scored])) if scored else None,
        "detected_only_delay_mean": float(np.mean([row["first_observed_alarm_delay"] for row in detected])) if detected else None,
        "detected_only_delay_count": len(detected), "unambiguous_delay_count": len(known_delay),
        "unambiguous_detected_delay_mean": float(np.mean([row["first_observed_alarm_delay"] for row in known_delay])) if known_delay else None}


def paired_span_rows(block, method, pairs):
    """Long table: one error and one matched normal row for each fixed pair."""
    rows = []
    for pair in pairs:
        for side in ("error", "normal"):
            start, end = pair[side + "_start"], pair[side + "_end"]
            rows.append(dict(pair, id=block["record"]["id"],
                source_id=block["record"]["source_id"], method=method, side=side,
                **_span_measurement(block, method, start, end)))
    return rows


def _profile_row(values, method, group, offset):
    scores = np.asarray([value[0] for value in values])
    alarms = np.asarray([value[1] for value in values], bool)
    finite = np.isfinite(scores)
    observed = int(finite.sum())
    return {"method": method, "group": group, "offset": offset, "eligible_tokens": len(scores),
        "scored_tokens": observed, "missing_tokens": len(scores) - observed,
        "score_mean": float(scores[finite].mean()) if observed else None,
        "alarms": int(alarms[finite].sum()),
        "alarm_rate": float(alarms[finite].mean()) if observed else None}


def offset_profile(blocks, method):
    """Gold-aligned descriptive curves, never inputs to the frozen detector."""
    offsets, phases = defaultdict(list), defaultdict(list)
    for block in blocks:
        for start, end in merge_token_spans(block["gold"]):
            middle = (end - start + 1) // 2
            for offset, position in enumerate(range(start, end)):
                value = (block["scores"][method][position], block["alarms"][method][position])
                offsets[offset].append(value)
                phases["front" if offset < middle else "back"].append(value)
                phases["onset" if offset == 0 else "continuation"].append(value)
    return {"offsets": [_profile_row(values, method, "offset", int(offset))
                         for offset, values in sorted(offsets.items())],
                "phases": [_profile_row(phases[name], method, name, None)
                        for name in ("onset", "continuation", "front", "back")]}
