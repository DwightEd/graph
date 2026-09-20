"""Finite head/source effects, conditional effects, adaptation and persistence."""

import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .cooperation import interaction_values
from .native import Intervention
from .scoring import evaluate_candidates


def unit_action(unit, dose=1., operation="cut", replacement=None, seed=0):
    return Intervention(unit["layer"], (), heads=(unit["head"],),
        queries=(unit["receiver"],), sources=tuple(unit["sources"]),
        dose=dose, operation=operation, replacement=replacement, seed=seed)


def measure_world(model, probe, actions=()):
    record, run = evaluate_candidates(model, probe, actions, sequence=probe["phase"] == "onset")
    record["measure"] = record["sequence_margin"] if probe["phase"] == "onset" else record["correct_first_logp"]
    record["readout"] = probe["readout"]
    logp = run.prefix_log_prob
    record["actual_logp"] = float(logp[probe["actual_token"]])
    record["entropy_nats"] = float(-(logp.exp() * logp).sum())
    if probe["phase"] == "onset":
        ids = sorted(set(record["candidate_first_ids"]))
        record["candidate_first_mass"] = float(logp[ids].exp().sum())
    return record, run


def baseline(model, probe, directory):
    path = directory / "baseline.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved["record"]))
            branches = saved["branches"].tolist()
            units = saved["units"].tolist()
            writes = {unit: {branch: torch.from_numpy(saved[branch][index].copy())
                            for branch in branches} for index, unit in enumerate(units)}
        return record, writes
    record, run = measure_world(model, probe)
    units = sorted(run.message_writes)
    arrays = {branch: np.stack([values[unit].float().numpy() for unit in units])
              for branch, values in run.branch_message_writes.items()}
    np.savez_compressed(path, record=json.dumps(record), units=np.array(units),
                        branches=np.array(list(arrays)), **arrays)
    writes = {unit: {branch: values[unit] for branch, values in run.branch_message_writes.items()}
              for unit in units}
    return record, writes


def cached_world(model, probe, actions, directory, name):
    path = directory / "worlds" / (name + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return json.loads(str(saved["record"]))
    record, run = measure_world(model, probe, actions)
    np.savez_compressed(path, record=json.dumps(record), changes_prefix=json.dumps(run.changes))
    return record


def single_effects(model, probe, units, doses, full, writes, directory, seed, atol):
    scores, rows, errors = {}, [], {}
    for unit in tqdm(units, desc=probe["phase"] + " head/source", leave=False):
        if not unit["sources"]:
            continue
        name = unit["unit"]
        restore = unit_action(unit, operation="replace", replacement=writes[name])
        sham = cached_world(model, probe, (restore,), directory, name + "_sham")
        errors[name] = abs(sham["measure"] - full["measure"])
        for dose in doses:
            cut = cached_world(model, probe, (unit_action(unit, dose),), directory, f"{name}_{dose:g}")
            scores[name, dose] = cut
            rows.append(dict(unit, dose=dose, full=full["measure"], cut=cut["measure"],
                support=full["measure"] - cut["measure"], sham_error=errors[name],
                numeric_ok=errors[name] <= atol, readout=probe["readout"]))
        if unit["source_group"] == "head_total":
            random = unit_action(unit, operation="random", seed=seed)
            score = cached_world(model, probe, (random,), directory, name + "_random")
            rows.append(dict(unit, dose=1., full=full["measure"], cut=score["measure"],
                support=full["measure"] - score["measure"], sham_error=errors[name],
                numeric_ok=errors[name] <= atol, readout=probe["readout"], control="equal_norm_random"))
    return scores, rows, errors


def conditional_effects(model, probe, units, pairs, doses, full, singles, errors, directory, atol):
    lookup = {unit["unit"]: unit for unit in units}
    rows = []
    for pair in tqdm(pairs, desc="four worlds", leave=False):
        left, right = lookup[pair["left"]], lookup[pair["right"]]
        if not left["sources"] or not right["sources"]:
            continue
        for dose in doses:
            actions = (unit_action(left, dose), unit_action(right, dose))
            name = f"{pair['left']}__{pair['right']}_{dose:g}"
            joint = cached_world(model, probe, actions, directory, name)
            without_left = singles[left["unit"], dose]["measure"]
            without_right = singles[right["unit"], dose]["measure"]
            values = interaction_values(full["measure"], without_left, without_right, joint["measure"])
            rows.append(dict(pair, dose=dose, **values, full=full["measure"],
                without_left=without_left, without_right=without_right, without_both=joint["measure"],
                left_conditional=without_right - joint["measure"],
                right_conditional=without_left - joint["measure"],
                sham_error=max(errors[left["unit"]], errors[right["unit"]]),
                numeric_ok=max(errors[left["unit"]], errors[right["unit"]]) <= atol,
                readout=probe["readout"]))
    return rows


def ordered_pairs(units, pairs, count):
    lookup = {unit["unit"]: unit for unit in units}
    result = []
    for pair in pairs:
        if pair["relation"] != "different_heads":
            continue
        early, late = sorted((lookup[pair["left"]], lookup[pair["right"]]), key=lambda row: row["layer"])
        if early["layer"] < late["layer"]:
            result.append((early, late))
    return result[:count]


def adaptation_effects(model, probe, units, pairs, full, singles, writes, directory, count, atol):
    rows = []
    for early, late in ordered_pairs(units, pairs, count):
        removed = singles[early["unit"], 1.]["measure"]
        for site in ("message", "mlp"):
            name = ("mlp_" if site == "mlp" else "") + late["unit"]
            restore = unit_action(late, operation="replace", replacement=writes[name])
            if site == "mlp":
                restore = Intervention(late["layer"], ("mlp",), queries=(late["receiver"],),
                                       operation="replace", replacement=writes[name])
            sham = cached_world(model, probe, (restore,), directory, name + "_sham")
            actions = (unit_action(early), restore)
            restored = cached_world(model, probe, actions, directory, early["unit"] + "_restore_" + name)
            error = abs(sham["measure"] - full["measure"])
            rows.append(dict(early=early["unit"], late=late["unit"], site=site,
                full=full["measure"], removed=removed, restored=restored["measure"],
                deletion_support=full["measure"] - removed,
                restoration_gain=restored["measure"] - removed,
                sham_error=error, numeric_ok=error <= atol, readout=probe["readout"]))
    return rows


def carried_effects(model, probe, onset_units, full, directory, atol):
    """Perturb only the earlier claim-onset receiver, then force the saved text."""
    if probe["phase"] not in ("back_half", "post_claim"):
        return []
    rows = []
    for unit in onset_units:
        if unit["source_group"] != "head_total":
            continue
        score = cached_world(model, probe, (unit_action(unit),), directory, unit["unit"] + "_at_onset")
        null = cached_world(model, probe, (unit_action(unit, 0.),), directory, unit["unit"] + "_onset_null")
        error = abs(null["measure"] - full["measure"])
        rows.append(dict(unit, full=full["measure"], changed=score["measure"],
            support=full["measure"] - score["measure"],
            null_error=error, numeric_ok=error <= atol,
            readout="later_observed_logp", token_path="fixed_teacher_forced"))
    return rows


def save_rows(rows, identity, directory, name):
    frame = pd.DataFrame(rows)
    for key, value in identity.items():
        frame[key] = value
    frame.to_csv(directory / (name + ".csv"), index=False)
