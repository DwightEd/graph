"""Hold the saved segmentation fixed and remove direct prior shrinkage from risk."""

import numpy as np
from state_audit.storage import (
    read_arrays,
    read_json,
    write_arrays,
    write_csv,
    write_json,
)

from .state_model import DIRECTORY as STATE_DIRECTORY
from .state_model import finish_evaluation, token_rows

DIRECTORY = "state_readout_v5"


def observed_readout(route, posterior):
    """Each posterior column n-1 averages the last n observations, including t."""
    prefix = np.r_[0., np.cumsum(route, dtype=np.float64)]
    score = np.empty(len(route))
    current_weight = np.empty(len(route))
    for target in range(len(route)):
        lengths = np.arange(1, target + 2)
        means = (prefix[target + 1] - prefix[target + 1 - lengths]) / lengths
        probability = posterior[target, :target + 1]
        score[target] = probability @ means
        current_weight[target] = probability @ (1 / lengths)
    return score, current_weight


def readout_rows(response, scores, methods):
    rows = token_rows(response, scores, methods)
    for target, row in enumerate(rows):
        row["prior_pull"] = float(scores["prior_pull"][target])
        row["observed_current_weight"] = float(scores["observed_current_weight"][target])
    return rows


def add_observed_readout(response, saved, baseline):
    if not np.array_equal(saved["token_id"], response["token_ids"][response["prompt_length"]:]):
        raise ValueError(f"{response['id']}: saved state token IDs differ from input")
    observed, weights = observed_readout(saved[baseline], saved["run_posterior"])
    saved.update(joint_observed=observed, observed_current_weight=weights,
                 prior_pull=saved["joint_state"] - observed)
    return saved


def run_readout(output, annotations=None, window=16):
    source = output / STATE_DIRECTORY / f"w{window}"
    destination = output / DIRECTORY / f"w{window}"
    summary = read_json(source / "scoring_protocol.json")
    summary.update(purpose="fixed_state_posterior_observed_readout", candidate_method="joint_observed",
                   score="posterior_weighted_empirical_segment_means_without_direct_prior_shrinkage",
                   segmentation_recomputed=False, original_state_directory=str(source),
                   comparison_controls=[summary["raw_baseline"], "route_mean", "joint_state"])
    summary["methods"]["joint_observed"] = "joint_observed"
    write_json(destination / "scoring_protocol.json", summary)
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(source / "responses" / f"{index:04d}" / "scores.npz")
        saved = add_observed_readout(response, saved, summary["raw_baseline"])
        current = readout_rows(response, saved, summary["methods"])
        directory = destination / "responses" / f"{index:04d}"
        write_arrays(directory / "scores.npz", **saved)
        write_csv(directory / "tokens.csv", current, list(current[0]))
        rows.extend(current)
    write_csv(destination / "tokens.csv", rows, list(rows[0]))
    summary["readout_diagnostics"] = {
        "mean_abs_prior_pull": float(np.mean([abs(row["prior_pull"]) for row in rows])),
        "reference_still_affects_segmentation": True,
    }
    return finish_evaluation(output, destination, summary, rows, annotations)


def evaluate_readout(output, annotations=None, window=16):
    destination = output / DIRECTORY / f"w{window}"
    summary = read_json(destination / "scoring_protocol.json")
    settings = read_json(output / "settings.json")
    rows = []
    for index, response in enumerate(settings["responses"]):
        saved = read_arrays(destination / "responses" / f"{index:04d}" / "scores.npz")
        rows.extend(readout_rows(response, saved, summary["methods"]))
    return finish_evaluation(output, destination, summary, rows, annotations)["evaluation"]
