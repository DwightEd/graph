"""Unlabelled, source-balanced empirical scales; ranks are not error probabilities."""

import numpy as np


def unit_values(values, units):
    return np.asarray([values[unit["start"]:unit["stop"]].mean() for unit in units])


def observations(record):
    units = record["views"]["units"]
    return dict(route=record["scores"]["raw_route"],
        local=unit_values(record["scores"]["source_local"], units),
        carrier=unit_values(record["scores"][record["carrier_score"]], units))


def fit_distribution(values, sources):
    """Each source has equal total weight, even with multiple long answers."""
    sources = np.asarray(sources)
    unique, counts = np.unique(sources, return_counts=True)
    source_count = dict(zip(unique, counts))
    weights = np.asarray([1 / (len(unique) * source_count[source]) for source in sources])
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values, dtype=float)[order]
    return dict(values=sorted_values, cumulative=np.r_[0., np.cumsum(weights[order])])


def fit_scales(records):
    measured = [observations(record) for record in records]
    result = {}
    for name in ("route", "local", "carrier"):
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
    raw = observations(record)
    calibrated = {name: transform(values, scales[name]) for name, values in raw.items()}
    unit_ids = record["scores"]["unit_id"]
    local, carrier = calibrated["local"][unit_ids], calibrated["carrier"][unit_ids]
    return dict(route=calibrated["route"], local_anchor=local, carrier_anchor=carrier,
                source_anchor=.5 * (local + carrier))
