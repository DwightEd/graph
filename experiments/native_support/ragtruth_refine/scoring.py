"""Separate ordering between source-score blocks from ordering inside them."""

import numpy as np

from ..dual_state.scoring import window_mean
from ..message_carriers.token_representation import aggregate_units
from ..ragtruth_benchmark.scoring import BASELINES
from ..unified.calibration import fit_distribution, transform

ANCHORS = ("local", "full", "pair")
CHANNELS = ("source_token", "source_window", "route_window")
WEIGHTS = (.025, .1, .25)
REFINEMENTS = ("tie", *(f"residual_{weight:g}" for weight in WEIGHTS))
CANDIDATES = tuple(f"{anchor}_{channel}_{mode}" for anchor in ANCHORS
                   for channel in CHANNELS for mode in REFINEMENTS)
METHODS = (*BASELINES, *CANDIDATES)
PRIMARY = "pair_source_window_tie"
SELECTION_POOL = ("source_local_unit_mean", "source_full_unit_mean", "source_pair_unit_mean",
                  "raw_route", "raw_route_offline_mean", *CANDIDATES)


def features(observed, units, window):
    sources = {name: observed[f"source_{name}"] for name in ("local", "full")}
    sources["pair"] = .5 * (sources["local"] + sources["full"])
    result = {"route_window": window_mean(observed["raw_route"], window, offline=True)}
    for name, values in sources.items():
        broadcast = aggregate_units(values, units)
        result[f"{name}_anchor"] = broadcast[[unit["start"] for unit in units]]
        result[f"{name}_source_token"] = values
        result[f"{name}_source_window"] = window_mean(values, window, offline=True)
    result["pair_anchor"] = .5 * (result["local_anchor"] + result["full_anchor"])
    return result


def fit_scales(rows):
    """Fit each task/split on unlabelled selected sources, with equal source weight."""
    result = {}
    for name in rows[0]["features"]:
        values = np.concatenate([row["features"][name] for row in rows])
        sources = np.concatenate([np.repeat(row["record"]["source_id"], len(row["features"][name]))
                                  for row in rows])
        result[name] = fit_distribution(values, sources)
    return result


def refine(anchor, observation_rank, units, anchor_scale):
    """Integer block spacing preserves every strict anchor comparison in tie mode."""
    residual = observation_rank - aggregate_units(observation_rank, units)
    levels = np.unique(anchor_scale["values"])
    block = np.searchsorted(levels, anchor).astype(float)
    # Residual is in [-1, 1]; width <= .5, while adjacent block centers are 1 apart.
    scores = {"tie": block + .25 * residual}
    anchor_rank = transform(anchor, anchor_scale)
    for weight in WEIGHTS:
        scores[f"residual_{weight:g}"] = anchor_rank + weight * residual
    return scores


def score_answer(row, scales):
    observed, units, measured = row["observed"], row["response"]["units"], row["features"]
    scores = {name: observed[name].copy() for name in ("source_local", "raw_route", "raw_attention", "entropy")}
    scores["raw_route_offline_mean"] = measured["route_window"]
    for anchor in ANCHORS:
        # Use the same floating-point order as source-first's saved pair baseline.
        if anchor == "pair":
            base = .5 * (scores["source_local_unit_mean"] + scores["source_full_unit_mean"])
        else:
            base = aggregate_units(observed[f"source_{anchor}"], units)
        scores[f"source_{anchor}_unit_mean"] = base
        for channel in CHANNELS:
            name = "route_window" if channel == "route_window" else f"{anchor}_{channel}"
            ranked = transform(measured[name], scales[name])
            for mode, values in refine(base, ranked, units, scales[f"{anchor}_anchor"]).items():
                scores[f"{anchor}_{channel}_{mode}"] = values
    if "previous_selected" in observed:
        scores["previous_selected"] = observed["previous_selected"].copy()
    return dict(token_id=observed["token_id"], **scores, risk=scores[PRIMARY].copy())
