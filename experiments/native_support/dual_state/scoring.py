"""Frozen directional ECDF and two temporal scales; no truth fitting or selection."""

import numpy as np

LAST_POSITION_BIN = 7


def position_bins(target):
    # The last bin is open ended; target answer length never enters causal scores.
    return np.minimum(np.floor(np.log2(np.asarray(target) + 1)).astype(int), LAST_POSITION_BIN)


def window_mean(values, window, offline=False):
    """Exactly window tokens internally, truncated at each answer's endpoints."""
    values = np.asarray(values, dtype=float)
    target = np.arange(len(values))
    past = (window - 1) // 2 if offline else window - 1
    future = window - 1 - past if offline else 0
    left, right = np.maximum(0, target - past), np.minimum(len(values), target + future + 1)
    cumulative = np.concatenate((np.zeros_like(values[:1]), np.cumsum(values, axis=0)), axis=0)
    count = (right - left).reshape((-1,) + (1,) * (values.ndim - 1))
    return (cumulative[right] - cumulative[left]) / count


def scalar_scores(record, window):
    scores = {}
    for name in ("raw_route", "observable_route"):
        current = record["baselines"][name]
        for mode in ("causal", "offline"):
            persistent = window_mean(current, window, offline=mode == "offline")
            scores[f"{name}_{mode}_mean"] = persistent
            scores[f"{name}_{mode}_dual"] = np.maximum(current, persistent)
    return scores


def reference_records(records, source_id):
    selected = [record for record in records if record["source_id"] != source_id]
    if len({record["source_id"] for record in selected}) < 2:
        raise ValueError("Head calibration needs at least two source-disjoint reference sources")
    return selected


def fit_reference(records):
    """Equal source weights within each position bin; empirical reference may contain errors."""
    route = np.concatenate([record["head_route"] for record in records])
    sources = np.concatenate([np.repeat(record["source_id"], record["response_length"]) for record in records])
    bins = np.concatenate([position_bins(record["target"]) for record in records])
    fitted = {}
    for position in np.unique(bins):
        selected = bins == position
        values, source_ids = route[selected], sources[selected]
        identities, counts = np.unique(source_ids, return_counts=True)
        weights_by_source = dict(zip(identities, 1 / counts))
        weights = np.asarray([weights_by_source[source] for source in source_ids])
        weights /= weights.sum()
        order = np.argsort(values, axis=0, kind="stable")
        sorted_values = np.take_along_axis(values, order, axis=0)
        cumulative = np.vstack((np.zeros(values.shape[1]), np.cumsum(weights[order], axis=0)))
        cumulative[-1] = 1.0
        fitted[int(position)] = dict(values=sorted_values, cumulative=cumulative,
                                     sources=identities.tolist(), tokens=len(values))
    return fitted


def calibrate(record, fitted):
    """2 * weighted mid-CDF - 1: positive means unusually history-directed for this head."""
    route = record["head_route"]
    bins = position_bins(record["target"])
    result = np.empty_like(route)
    for position in np.unique(bins):
        if int(position) not in fitted:
            raise ValueError(f"No reference observations in position bin {position}")
        cell = fitted[int(position)]
        selected = bins == position
        for head in range(route.shape[1]):
            values, cumulative = cell["values"][:, head], cell["cumulative"][:, head]
            left = np.searchsorted(values, route[selected, head], side="left")
            right = np.searchsorted(values, route[selected, head], side="right")
            result[selected, head] = cumulative[left] + cumulative[right] - 1
    # No measurable output response is not evidence for a directed deviation.
    result[record["response_energy"] == 0] = 0
    return result


def positive_rms(values):
    return np.sqrt(np.mean(np.maximum(values, 0) ** 2, axis=-1))


def head_scores(record, fitted, window):
    state = calibrate(record, fitted)
    current = positive_rms(state)
    scores = dict(head_uncalibrated_mean=record["head_route"].mean(1), head_current=current)
    channels = dict(current=state)
    for mode in ("causal", "offline"):
        persistent = window_mean(state, window, offline=mode == "offline")
        channels[f"{mode}_persistent"] = persistent
        scores[f"head_{mode}_persistent"] = positive_rms(persistent)
        scores[f"head_{mode}_dual"] = np.maximum(current, scores[f"head_{mode}_persistent"])
    return scores, channels
