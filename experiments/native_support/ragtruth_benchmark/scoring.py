"""Source-first ranking with a small, predeclared route-shrinkage family."""

from collections import defaultdict

import numpy as np
from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from ..dual_state.scoring import window_mean
from ..message_carriers.token_representation import aggregate_units
from ..unified.calibration import fit_distribution, transform, unit_values

WEIGHTS = (0., .1, .25, .5, 1.)
FUSIONS = tuple(f"local_route_{weight:g}" for weight in WEIGHTS)
PRIMARY = "source_local_unit_mean"
BASELINES = (PRIMARY, "source_local", "source_full_unit_mean", "source_pair_unit_mean",
             "raw_route", "raw_route_offline_mean", "raw_attention", "entropy")
METHODS = (*BASELINES, *FUSIONS)


def score_answer(observed, response, scales, window):
    units = response["units"]
    scores = {name: value.copy() for name, value in observed.items()}
    for name in ("source_local", "source_full"):
        scores[name + "_unit_mean"] = aggregate_units(observed[name], units)
    scores["source_pair_unit_mean"] = .5 * (scores["source_local_unit_mean"] + scores["source_full_unit_mean"])
    scores["raw_route_offline_mean"] = window_mean(observed["raw_route"], window, offline=True)
    source_rank = transform(scores[PRIMARY], scales["source"])
    route_rank = transform(observed["raw_route"], scales["route"])
    residual = route_rank - aggregate_units(route_rank, units)
    for method, weight in zip(FUSIONS, WEIGHTS):
        scores[method] = source_rank + weight * residual
    scores.update(target=np.arange(len(source_rank)), risk=scores[PRIMARY].copy())
    components = dict(source_rank=source_rank, route_rank=route_rank, route_residual=residual)
    return scores, components


def fit_group(output, records):
    source_values, source_ids, route_values, route_ids = [], [], [], []
    for record in records:
        directory = output / record["directory"]
        values, response = read_arrays(directory / "observations.npz"), read_json(directory / "response.json")
        if not np.array_equal(values["token_id"], response["answer_ids"]):
            raise ValueError(f"{record['id']}: measurement and original tokens differ")
        if not all(np.isfinite(values[name]).all() for name in ("source_local", "source_full", "raw_route", "raw_attention", "entropy")):
            raise ValueError(f"{record['id']}: incomplete/nonfinite observations")
        local = unit_values(values["source_local"], response["units"])
        source_values.extend(local)
        source_ids.extend([record["source_id"]] * len(local))
        route_values.extend(values["raw_route"])
        route_ids.extend([record["source_id"]] * len(values["raw_route"]))
    return dict(source=fit_distribution(source_values, source_ids), route=fit_distribution(route_values, route_ids))


def score_all(args, manifest):
    groups = defaultdict(list)
    for record in manifest["records"]:
        groups[record["task"], record["split"]].append(record)
    for (task, split), records in groups.items():
        scales = fit_group(args.output, records)
        for name, distribution in scales.items():
            write_arrays(args.output / "scales" / task / split / f"{name}.npz", **distribution)
        for record in tqdm(records, desc=f"freeze {task}/{split} token scores"):
            directory = args.output / record["directory"]
            values = read_arrays(directory / "observations.npz")
            scores, components = score_answer(values, read_json(directory / "response.json"), scales, args.window)
            write_arrays(directory / "scores.npz", **scores)
            write_arrays(directory / "components.npz", **components)
    coverage = dict(status="complete", selected_answers=manifest["selected_answers"],
        scored_answers=len(manifest["records"]), scored_tokens=sum(r["tokens"] for r in manifest["records"]),
        excluded=manifest["excluded"], truncated_tokens=0, labels_used_for_scoring=False)
    write_json(args.output / "coverage.json", coverage)
    return coverage
