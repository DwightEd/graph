"""Unlabelled, source-balanced empirical scales; ranks are not error probabilities."""

import numpy as np


def unit_values(values, units):
    return np.asarray([values[unit["start"]:unit["stop"]].mean() for unit in units])


def observations(record, include_tokens=False):
    units = record["views"]["units"]
    measured = dict(route=record["scores"]["raw_route"],
        local=unit_values(record["scores"]["source_local"], units),
        carrier=unit_values(record["scores"][record["carrier_score"]], units))
    if include_tokens:
        measured.update(local_token=record["scores"]["source_local"],
                        carrier_token=record["scores"][record["carrier_score"]])
    return measured


def fit_distribution(values, sources):
    """Each source has equal total weight, even with multiple long answers."""
    sources = np.asarray(sources)
    unique, counts = np.unique(sources, return_counts=True)
    source_count = dict(zip(unique, counts))
    weights = np.asarray([1 / (len(unique) * source_count[source]) for source in sources])
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    return dict(values=sorted_values, cumulative=np.r_[0., np.cumsum(weights[order])])


def fit_scales(records, include_tokens=False):
    measured = [observations(record, include_tokens) for record in records]
    result = {}
    for name in measured[0]:
        values = np.concatenate([row[name] for row in measured])
        sources = np.concatenate([np.repeat(record["response"]["source_id"], len(row[name]))
                                  for record, row in zip(records, measured)])
        result[name] = fit_distribution(values, sources)
    return result


def transform(values, distribution):
    left = np.searchsorted(distribution["values"], values, side="left")
    right = np.searchsorted(distribution["values"], values, side="right")
    cumulative = distribution["cumulative"]
    return .5 * (cumulative[left] + cumulative[right])


def channels(record, scales):
    raw = observations(record, include_tokens="local_token" in scales)
    calibrated = {name: transform(values, scales[name]) for name, values in raw.items()}
    unit_ids = record["scores"]["unit_id"]
    local, carrier = calibrated["local"][unit_ids], calibrated["carrier"][unit_ids]
    result = dict(route=calibrated["route"], local_anchor=local, carrier_anchor=carrier,
                  source_anchor=.5 * (local + carrier))
    if "local_token" in calibrated:
        result.update(local_token=calibrated["local_token"], carrier_token=calibrated["carrier_token"])
        result["source_token"] = .5 * (result["local_token"] + result["carrier_token"])
        result["token_observation"] = .5 * (result["source_token"] + result["route"])
    return result
