"""Causally aligned events and within-response controls."""

from __future__ import annotations

import numpy as np

from .claims import FORCED_CHUNK


def validate_coordinates(result, token_ids, response_start: int, sample_id: str) -> int:
    prediction = np.asarray(result["prediction_position"], dtype=np.int64)
    query = np.asarray(result["query_position"], dtype=np.int64)
    target = np.asarray(result["target_token_id"], dtype=np.int64)
    if not np.array_equal(prediction, response_start + np.arange(len(prediction))):
        raise ValueError(f"prediction coordinates changed: {sample_id}")
    if not np.array_equal(query + 1, prediction):
        raise ValueError(f"query/target alignment is not q=p-1: {sample_id}")
    if not np.array_equal(target, np.asarray(token_ids)[prediction]):
        raise ValueError(f"target tokens changed: {sample_id}")
    starts = np.asarray(result["claim_start"], dtype=np.int64)
    stops = np.asarray(result["claim_stop"], dtype=np.int64)
    valid = (
        starts.shape == stops.shape
        and np.all(starts >= response_start)
        and np.all(starts < stops)
        and np.all(stops <= response_start + len(prediction))
    )
    if not valid:
        raise ValueError(f"claim coordinates are invalid: {sample_id}")
    return len(prediction)


def positive_onsets(label) -> np.ndarray:
    label = np.asarray(label, dtype=bool)
    return np.flatnonzero(label & ~np.r_[False, label[:-1]])


def pulse(series, center: int, pre: int, post: int) -> float:
    """Post-window minus pre-window; descriptive for offsets after ``center``."""

    series = np.asarray(series, dtype=np.float64)
    if center < pre or center + post > len(series):
        return float("nan")
    before = series[center - pre : center]
    after = series[center : center + post]
    if not (np.isfinite(before).all() and np.isfinite(after).all()):
        return float("nan")
    return float(after.mean() - before.mean())


def entry_change(series, center: int, pre: int) -> float:
    """Value at p=center minus its past baseline.

    The query for the center event is q=p-1, so this statistic cannot read the
    token being predicted or any later hallucinated token.
    """

    series = np.asarray(series, dtype=np.float64)
    if center < pre or center >= len(series):
        return float("nan")
    before = series[center - pre : center]
    current = series[center]
    if not (np.isfinite(before).all() and np.isfinite(current)):
        return float("nan")
    return float(current - before.mean())


def aligned_change(series, center: int, low: int, high: int, *, reverse=False):
    """Complete visualization window, independent of scalar event inclusion."""

    series = np.asarray(series, dtype=np.float64)
    if center + low < 0 or center + high >= len(series) or low >= 0:
        return None
    values = series[center + np.arange(low, high + 1)]
    baseline = series[center + low : center]
    if not (np.isfinite(values).all() and np.isfinite(baseline).all()):
        return None
    change = values - baseline.mean()
    return -change if reverse else change


def stage_series(row: dict, kind: str) -> dict[str, np.ndarray]:
    trace = row.get("functional_log_lift_trace")
    if trace is None:
        return {}
    if kind == "evidence_specificity":
        series = trace[:, :, 0] - trace[:, :, 1]
    elif kind == "history_enrichment":
        series = trace[:, :, 2]
    else:
        raise ValueError(f"unknown stage series: {kind}")
    bands = np.array_split(np.arange(series.shape[0]), 3)
    return {
        name: series[band].mean(axis=0)
        for name, band in zip(("early", "middle", "late"), bands, strict=True)
        if len(band)
    }


def event_features(
    row: dict,
    center: int,
    *,
    pre: int,
    post: int,
    curve_low: int,
    curve_high: int,
) -> dict | None:
    """Return scalar features whenever their short window is complete.

    Long event curves are optional. This prevents a plotting choice such as
    ``curve_high=10`` from silently changing the hypothesis-test sample.
    """

    evidence_entry = entry_change(row["evidence_specificity"], center, pre)
    history_entry = -entry_change(row["history_enrichment"], center, pre)
    evidence_pulse = pulse(row["evidence_specificity"], center, pre, post)
    history_release = -pulse(row["history_enrichment"], center, pre, post)
    if not np.isfinite(evidence_entry):
        return None

    result = {
        "center": int(center),
        "evidence_entry": evidence_entry,
        "history_entry_release": history_entry,
        "evidence_post_pulse": evidence_pulse,
        "history_post_release": history_release,
        "evidence_curve": aligned_change(
            row["evidence_specificity"], center, curve_low, curve_high
        ),
        "history_curve": aligned_change(
            row["history_enrichment"], center, curve_low, curve_high, reverse=True
        ),
    }
    for stage, series in stage_series(row, "evidence_specificity").items():
        result[f"{stage}_evidence_entry"] = entry_change(series, center, pre)
    for stage, series in stage_series(row, "history_enrichment").items():
        result[f"{stage}_history_entry_release"] = -entry_change(series, center, pre)
    for name in ("direct_evidence_cut_delta", "global_evidence_cut_delta"):
        values = row.get(name)
        if values is not None and center < len(values):
            result[name.removesuffix("_delta") + "_dependence"] = float(-values[center])
    return result


def average_event_features(events: list[dict]) -> dict | None:
    if not events:
        return None
    result: dict[str, object] = {"center": int(np.median([event["center"] for event in events]))}
    for name in events[0]:
        if name == "center":
            continue
        values = [
            event[name]
            for event in events
            if event.get(name) is not None
            and np.isfinite(np.asarray(event[name], dtype=np.float64)).all()
        ]
        if not values:
            result[name] = None
            continue
        stacked = np.asarray(values, dtype=np.float64)
        result[name] = (
            float(stacked.mean()) if stacked.ndim == 1 else stacked.mean(axis=0)
        )
    return result


def boundary_events(
    row: dict,
    *,
    pre: int,
    post: int,
    curve_low: int,
    curve_high: int,
) -> list[dict]:
    """Natural sentence-boundary proxies; forced length chunks are excluded."""

    events = []
    onsets = positive_onsets(row["label"])
    response_start = row["response_start"]
    kinds = row.get("claim_boundary_kind")
    if kinds is None:
        kinds = np.ones(len(row["claim_start"]), dtype=np.int64)
    for absolute_start, absolute_stop, kind in zip(
        row["claim_start"], row["claim_stop"], kinds, strict=True
    ):
        if int(kind) == FORCED_CHUNK:
            continue
        start = int(absolute_start) - response_start
        stop = int(absolute_stop) - response_start
        boundary = event_features(
            row,
            start,
            pre=pre,
            post=post,
            curve_low=curve_low,
            curve_high=curve_high,
        )
        if boundary is None:
            continue

        control_low, control_high = start + pre + post, stop - post
        control_positions: list[int] = []
        if control_low <= control_high:
            requested = np.linspace(
                control_low, control_high, min(3, control_high - control_low + 1)
            ).round().astype(int)
            control_positions = sorted(set(requested.tolist()))
        controls = [
            event
            for position in control_positions
            if (
                event := event_features(
                    row,
                    position,
                    pre=pre,
                    post=post,
                    curve_low=curve_low,
                    curve_high=curve_high,
                )
            )
            is not None
        ]
        control = average_event_features(controls)

        claim_onsets = onsets[(onsets >= start) & (onsets < stop)]
        onset_offset = None if not len(claim_onsets) else int(claim_onsets[0] - start)
        local_pre_clean = not bool(row["label"][max(0, start - pre) : start].any())
        prefix_clean = not bool(row["label"][:start].any())
        correct = prefix_clean and not bool(row["label"][start:stop].any())
        exact = onset_offset == 0
        near = onset_offset is not None and 0 < onset_offset < post
        events.append(
            {
                "source_id": row["source_id"],
                "sample_id": row["sample_id"],
                "correct": correct,
                "local_pre_clean": local_pre_clean,
                "prefix_clean": prefix_clean,
                "onset_offset": onset_offset,
                "onset_at_boundary": exact,
                "onset_near_boundary": near,
                "late_onset": onset_offset is not None and not (exact or near),
                "preceding_token_id": (
                    int(row["target_token_id"][start - 1]) if start > 0 else None
                ),
                "boundary": boundary,
                "control": control,
                "control_positions": control_positions,
            }
        )
    return events


def onset_class(center: int, boundaries: np.ndarray, post: int) -> str:
    preceding = boundaries[boundaries <= center]
    if not len(preceding):
        return "unassigned"
    offset = int(center - preceding[-1])
    if offset == 0:
        return "at_boundary"
    return "near_boundary" if offset < post else "late"


def onset_pairs(
    row: dict,
    *,
    pre: int,
    post: int,
    curve_low: int,
    curve_high: int,
    position_caliper: int = 32,
) -> list[dict]:
    """Hallucination onsets matched to clean tokens in the same response."""

    label = np.asarray(row["label"], dtype=bool)
    token = np.asarray(row["target_token_id"])
    starts = np.asarray(row["claim_start"])
    kinds = np.asarray(
        row.get("claim_boundary_kind", np.ones(len(starts))), dtype=np.int64
    )
    boundaries = starts[kinds != FORCED_CHUNK] - row["response_start"]
    candidates = [
        center
        for center in range(pre, len(label) - post + 1)
        if not label[center - pre : center + post].any()
        and not np.any(np.abs(boundaries - center) < post)
    ]
    used, pairs = set(), []
    for center in positive_onsets(label):
        positive = event_features(
            row,
            int(center),
            pre=pre,
            post=post,
            curve_low=curve_low,
            curve_high=curve_high,
        )
        available = [
            candidate
            for candidate in candidates
            if candidate not in used and abs(candidate - center) <= position_caliper
        ]
        if positive is None or not available:
            continue
        exact = [
            candidate
            for candidate in available
            if token[candidate] == token[center]
            and abs(candidate - center) <= position_caliper
        ]
        control_center = min(
            exact or available,
            key=lambda candidate: (abs(candidate - center), candidate),
        )
        control = event_features(
            row,
            control_center,
            pre=pre,
            post=post,
            curve_low=curve_low,
            curve_high=curve_high,
        )
        if control is None:
            continue
        used.add(control_center)
        pairs.append(
            {
                "source_id": row["source_id"],
                "sample_id": row["sample_id"],
                "onset_class": onset_class(int(center), boundaries, post),
                "token_matched": bool(exact),
                "position_distance": int(abs(control_center - center)),
                "positive": positive,
                "control": control,
            }
        )
    return pairs
