"""Finite message deletions, four worlds, and controlled downstream restoration."""

import json

import numpy as np
import torch
from tqdm import tqdm

from .cooperation import interaction_values
from .native import Intervention
from .scoring import evaluate_candidates
from .transport_plan import causal_order


METRICS = ("next_margin", "sequence_margin")


def message_action(unit, dose=1., operation="cut", replacement=None, seed=0):
    return Intervention(layer=unit["layer"], groups=(), heads=(unit["head"],),
                        queries=(unit["receiver"],), sources=(unit["source"],),
                        dose=dose, operation=operation, replacement=replacement, seed=seed)


def baseline_world(model, probe, directory):
    path = directory / "baseline.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            score = json.loads(str(saved["record"]))
            writes = {str(unit): torch.from_numpy(vector.copy())
                      for unit, vector in zip(saved["units"], saved["writes"])}
            reference = torch.from_numpy(saved["prefix_logp"].copy())
        return score, writes, reference
    score, run = evaluate_candidates(model, probe)
    units = sorted(run.message_writes)
    writes = np.stack([run.message_writes[unit].float().numpy() for unit in units])
    np.savez_compressed(path, record=json.dumps(score), units=np.array(units), writes=writes,
                        prefix_logp=run.prefix_log_prob.numpy())
    return score, run.message_writes, run.prefix_log_prob


def run_world(model, probe, directory, name, actions, reference):
    path = directory / "worlds" / (name + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return json.loads(str(saved["record"]))
    score, run = evaluate_candidates(model, probe, actions, reference_logp=reference)
    np.savez_compressed(path, record=json.dumps(score), trajectory=json.dumps(run.trajectory),
                        changes=json.dumps(run.changes))
    return score


def single_worlds(model, probe, units, doses, baseline, reference, directory, seed):
    scores, rows = {}, []
    for unit in tqdm(units, desc="message deletions", leave=False):
        for dose in doses:
            name = f"{unit['unit']}_cut_{dose:g}"
            action = message_action(unit, dose)
            score = run_world(model, probe, directory, name, (action,), reference)
            scores[unit["unit"], dose] = score
            rows.append(dict(unit, dose=dose, treatment="cut", **score,
                             first_support=baseline["next_margin"] - score["next_margin"],
                             sequence_support=baseline["sequence_margin"] - score["sequence_margin"],
                             linear_prediction=dose * unit["final_linear_support"]))
        action = message_action(unit, operation="random", seed=seed)
        score = run_world(model, probe, directory, unit["unit"] + "_random", (action,), reference)
        rows.append(dict(unit, dose=1., treatment="random_direction", **score,
                         first_support=baseline["next_margin"] - score["next_margin"],
                         sequence_support=baseline["sequence_margin"] - score["sequence_margin"]))
    return scores, rows


def joint_worlds(model, probe, units, pairs, doses, baseline, singles, reference, directory):
    lookup = {row["unit"]: row for row in units}
    rows = []
    for pair in tqdm(pairs, desc="head interactions", leave=False):
        left, right = lookup[pair["left"]], lookup[pair["right"]]
        for dose in doses:
            actions = (message_action(left, dose), message_action(right, dose))
            name = f"{pair['left']}_{pair['right']}_cut_{dose:g}"
            joint = run_world(model, probe, directory, name, actions, reference)
            a, b = singles[left["unit"], dose], singles[right["unit"], dose]
            for metric in METRICS:
                values = interaction_values(baseline[metric], a[metric], b[metric], joint[metric])
                rows.append(dict(pair, dose=dose, metric=metric, **values,
                                 full=baseline[metric], without_left=a[metric],
                                 without_right=b[metric], without_both=joint[metric],
                                 left_when_right_absent=b[metric] - joint[metric],
                                 right_when_left_absent=a[metric] - joint[metric]))
    return rows


def restored_worlds(model, probe, early, late, site, writes, reference, directory):
    prefix = "mlp_" if site == "mlp" else ""
    replacement = writes[prefix + late["unit"]]
    if site == "mlp":
        restore = Intervention(late["layer"], ("mlp",), queries=(late["receiver"],),
                               operation="replace", replacement=replacement)
    else:
        restore = message_action(late, operation="replace", replacement=replacement)
    name = f"{early['unit']}_restore_{site}_{late['unit']}"
    actions = (message_action(early), restore)
    restored = run_world(model, probe, directory, name, actions, reference)
    sham = run_world(model, probe, directory, f"sham_{site}_{late['unit']}", (restore,), reference)
    return restored, sham


def mediation_worlds(model, probe, units, pairs, baseline, singles, writes, reference, directory):
    lookup = {row["unit"]: row for row in units}
    rows = []
    for pair in tqdm(pairs, desc="downstream restoration", leave=False):
        order = causal_order(lookup[pair["left"]], lookup[pair["right"]])
        if order is None:
            continue
        early, late = order
        removed = singles[early["unit"], 1.]
        for site in ("message", "mlp"):
            restored, sham = restored_worlds(
                model, probe, early, late, site, writes, reference, directory)
            for metric in METRICS:
                rows.append(dict(early=early["unit"], late=late["unit"], site=site,
                                 metric=metric, full=baseline[metric], cut=removed[metric],
                                 restored=restored[metric], restoration_gain=restored[metric] - removed[metric],
                                 sham_delta=sham[metric] - baseline[metric],
                                 deletion_support=baseline[metric] - removed[metric]))
    return rows
