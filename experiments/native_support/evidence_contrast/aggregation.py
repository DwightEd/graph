"""One frozen text partition for every contrast and control; no label-dependent scores."""

import numpy as np

TOKEN_METHODS = ("source_full", "source_local", "source_pair", "raw_route",
                 "observable_route", "raw_attention", "entropy", "surprisal_full")
UNIT_METHODS = tuple(f"{name}_unit_mean" for name in TOKEN_METHODS)
WINDOW_CONTROLS = ("raw_route_offline_mean", "observable_route_offline_mean")
METHODS = (*TOKEN_METHODS, *UNIT_METHODS, *WINDOW_CONTROLS)
COMPARISONS = tuple(
    (f"{name}_unit_mean", control)
    for name in ("source_full", "source_local", "source_pair")
    for control in (name, "raw_route_unit_mean", "observable_route_unit_mean", *WINDOW_CONTROLS)
)


def aggregate_scores(saved, views):
    """Broadcast unit means while retaining each original token score and default risk."""
    count = len(views["answer_ids"])
    covered = [t for unit in views["units"] for t in range(unit["start"], unit["stop"])]
    if covered != list(range(count)):
        raise ValueError("Saved units must partition the original answer in order")
    if not np.array_equal(saved["token_id"], views["answer_ids"]):
        raise ValueError("Saved score and unit target identities differ")
    if not np.array_equal(saved["target"], np.arange(count)):
        raise ValueError("Saved scores must cover every original target")
    result = {name: saved[name] for name in (*TOKEN_METHODS, *WINDOW_CONTROLS, "risk", "target", "token_id")}
    for name in TOKEN_METHODS:
        values = np.asarray(saved[name], dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete {name} observations")
        mean = np.empty(count)
        for unit in views["units"]:
            selected = slice(unit["start"], unit["stop"])
            mean[selected] = values[selected].mean()
        result[f"{name}_unit_mean"] = mean
    result["unit_id"] = np.empty(count, dtype=np.int64)
    result["unit_end_target"] = np.empty(count, dtype=np.int64)
    for index, unit in enumerate(views["units"]):
        selected = slice(unit["start"], unit["stop"])
        result["unit_id"][selected] = index
        result["unit_end_target"][selected] = unit["stop"] - 1
    return result


def select_whole_units(scores, fraction=.1):
    """Keep boundary ties together; report actual cost instead of splitting tied units."""
    values = np.asarray(scores)
    if not len(values):
        return np.zeros(0, dtype=bool)
    count = max(1, int(np.ceil(len(values) * fraction)))
    threshold = np.sort(values)[-count]
    return values >= threshold
