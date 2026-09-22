"""Source-disjoint route-volume calibration; historical and causal protocols stay named."""

import hashlib
from collections import Counter

import numpy as np

PROTOCOLS = {
    "functional_collapse": ("functional_ordinary_log_volume", "causal"),
    "attention_collapse": ("attention_ordinary_log_volume", "causal"),
    "functional_collapse_offline": ("functional_legacy_log_volume", "historical_offline"),
    "attention_collapse_offline": ("attention_legacy_log_volume", "historical_offline"),
}


def source_partitions(records, folds=5, seed=0):
    sources = {record["source_id"] for record in records}
    if len(sources) < 3:
        return []
    def key(source):
        return int.from_bytes(hashlib.sha256(f"{seed}\0{source}".encode()).digest()[:8], "big")
    ordered = sorted(sources, key=lambda source: (key(source), source))
    count = min(folds, len(ordered))
    assignment = {source: index % count for index, source in enumerate(ordered)}
    return [
        {"test": [s for s in ordered if assignment[s] == fold],
         "calibration": [s for s in ordered if assignment[s] == (fold + 1) % count],
         "fit": [s for s in ordered if assignment[s] not in (fold, (fold + 1) % count)]}
        for fold in range(count)
    ]


def design(prompt_length, tokens, protocol):
    position = np.arange(tokens, dtype=float)
    if protocol == "historical_offline":
        position = (position + 0.5) / tokens
        length = np.full(tokens, np.log1p(prompt_length + tokens))
    else:
        position = np.log1p(position)
        length = np.full(tokens, np.log1p(prompt_length))
    return np.column_stack((np.ones(tokens), position, position ** 2, length))


def position_bins(tokens, protocol):
    position = np.arange(tokens)
    if protocol == "historical_offline":
        return np.minimum(position * 10 // tokens, 9)
    return np.minimum(np.floor(np.log2(position + 1)).astype(int), 9)


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return values[order[np.searchsorted(cumulative, cumulative[-1] / 2)]]


def record_weights(records):
    counts = Counter(record["source_id"] for record in records)
    return {
        record["id"]: np.full(record["tokens"], 1 / counts[record["source_id"]] / record["tokens"])
        for record in records
    }


def fit_volume(records, field, protocol):
    weights = record_weights(records)
    weight = np.concatenate([weights[r["id"]] for r in records])
    inputs = np.concatenate([design(r["prompt_length"], r["tokens"], protocol) for r in records])
    volumes = np.concatenate([r[field] for r in records])
    root = np.sqrt(weight)
    coefficients = np.linalg.lstsq(inputs * root[:, None], volumes * root[:, None], rcond=None)[0]
    residual = volumes - inputs @ coefficients
    scale = []
    for values in residual.T:
        center = weighted_median(values, weight)
        scale.append(max(1.4826 * weighted_median(abs(values - center), weight), 1e-3))
    return coefficients, np.asarray(scale)


def raw_collapse(record, field, protocol, model):
    coefficients, scale = model
    expected = design(record["prompt_length"], record["tokens"], protocol) @ coefficients
    return np.maximum((expected - record[field]) / scale, 0).mean(-1)


def calibration_tables(records, scores, protocol):
    weights = record_weights(records)
    values = np.concatenate([scores[r["id"]] for r in records])
    weight = np.concatenate([weights[r["id"]] for r in records])
    buckets = np.concatenate([position_bins(r["tokens"], protocol) for r in records])
    tables = {}
    for bucket in range(-1, 10):
        selected = buckets == bucket
        if bucket == -1 or not selected.any():
            selected = np.ones(len(values), dtype=bool)
        order = np.argsort(values[selected], kind="stable")
        cumulative = np.cumsum(weight[selected][order])
        tables[bucket] = values[selected][order], cumulative / cumulative[-1]
    return tables


def calibrated_score(record, field, protocol, model, tables):
    raw = raw_collapse(record, field, protocol, model)
    result = np.zeros(len(raw))
    for index, bucket in enumerate(position_bins(len(raw), protocol)):
        reference, cumulative = tables[int(bucket)]
        location = np.searchsorted(reference, raw[index], side="right")
        if location:
            result[index] = cumulative[location - 1]
    return result


def crossfit_collapse(records):
    scores = {}
    for record in records:
        scores[record["id"]] = {name: np.full(record["tokens"], np.nan) for name in PROTOCOLS}
    partitions = source_partitions(records)
    fitted = []
    for partition in partitions:
        fit = [r for r in records if r["source_id"] in partition["fit"]]
        calibration = [r for r in records if r["source_id"] in partition["calibration"]]
        test = [r for r in records if r["source_id"] in partition["test"]]
        models = {}
        for name, (field, protocol) in PROTOCOLS.items():
            model = fit_volume(fit, field, protocol)
            raw = {r["id"]: raw_collapse(r, field, protocol, model) for r in calibration}
            tables = calibration_tables(calibration, raw, protocol)
            for record in test:
                scores[record["id"]][name] = calibrated_score(record, field, protocol, model, tables)
            models[name] = {"coefficients": model[0].tolist(), "scales": model[1].tolist()}
        fitted.append({**partition, "models": models})
    return scores, {
        "status": "available" if partitions else "unavailable",
        "reason": None if partitions else "at_least_three_distinct_sources_required",
        "labels_used": False, "seed": 0, "partitions": fitted,
        "historical_offline_uses_completed_answer_length": True,
        "causal_uses_only_prompt_length_and_current_position": True,
    }
