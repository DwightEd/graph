"""Cached native cuts, branch-matched restoration, and cross-position checks."""

import json

import pandas as pd
from tqdm import tqdm

from ..path_conflict.native import Intervention
from ..path_conflict.paired_trials import baseline, cached_world, unit_action
from ..path_conflict.paired_exports import export_record


def world(model, probe, directory, name, actions=()):
    record = cached_world(model, probe, actions, directory, name)
    (directory / "worlds" / (name + ".json")).write_text(json.dumps(record, indent=2, allow_nan=False))
    return record["measure"]


def restore_message(unit, writes):
    return unit_action(unit, operation="replace", replacement=writes[unit["unit"]])


def restore_state(unit, writes):
    return Intervention(unit["layer"], ("residual",), queries=(unit["receiver"],),
                        operation="replace", replacement=writes["residual_" + unit["unit"]])


def event_controls(model, probe, directory, event, full, writes):
    name = event["entry"]["unit"]
    restore = restore_state(event["entry"], writes)
    state_sham = world(model, probe, directory, name + "_state_sham", (restore,))
    result = dict(state_sham_error=abs(state_sham - full), restore_state=restore)
    if event["relay"] is not None:
        relay = event["relay"]
        restored = restore_message(relay, writes)
        result["relay_sham_error"] = abs(world(model, probe, directory, name + "_relay_sham", (restored,)) - full)
        result["without_relay"] = world(model, probe, directory, name + "_cut_relay", (unit_action(relay),))
        result["restore_relay"] = restored
    return result


def relay_worlds(model, probe, directory, name, event, cut, full, removed, controls):
    relay_cut = unit_action(event["relay"])
    joint = world(model, probe, directory, name + "_cut_both", (cut, relay_cut))
    restored = world(model, probe, directory, name + "_restore_state", (cut, controls["restore_state"]))
    blocked = world(model, probe, directory, name + "_restore_state_blocked",
                    (cut, controls["restore_state"], relay_cut))
    restored_message = world(model, probe, directory, name + "_restore_relay", (cut, controls["restore_relay"]))
    without_relay = controls["without_relay"]
    gain = restored - removed
    blocked_gain = blocked - joint
    return dict(without_relay=without_relay, without_both=joint,
        entry_conditional=without_relay - joint, relay_support=full - without_relay,
        relay_conditional=removed - joint,
        interaction=full - removed - without_relay + joint,
        restored_state=restored, restored_state_blocked=blocked,
        restoration_identity_error=abs(restored - full),
        blocked_identity_error=abs(blocked - without_relay),
        restoration_gain=gain, blocked_restoration_gain=blocked_gain, gate_difference=gain - blocked_gain,
        restored_relay=restored_message, relay_restoration_gain=restored_message - removed)


def unit_trials(model, probe, directory, event, unit, doses, full, writes, controls, atol):
    restore = restore_message(unit, writes)
    sham = world(model, probe, directory, unit["unit"] + "_sham", (restore,))
    errors = dict(entry_sham_error=abs(sham - full), state_sham_error=controls["state_sham_error"])
    if event["relay"] is not None:
        joint_sham = world(model, probe, directory, unit["unit"] + "_joint_sham",
                           (restore, controls["restore_relay"]))
        errors.update(relay_sham_error=controls["relay_sham_error"], joint_sham_error=abs(joint_sham - full))
    error = max(errors.values())
    rows = []
    for dose in doses:
        name = unit["unit"] + f"_{dose:g}"
        cut = unit_action(unit, dose)
        removed = world(model, probe, directory, name, (cut,))
        row = dict(event=event["event"], selection=event["selected"]["selection"],
            **unit, **errors, dose=dose, relay_dose=1., full=full, removed=removed, support=full - removed,
            sham_error=error, numeric_ok=error <= atol, relay_status=event["relay_status"])
        if event["relay"] is not None:
            row.update(relay_worlds(model, probe, directory, name, event, cut, full, removed, controls))
            row["numeric_ok"] &= max(row["restoration_identity_error"], row["blocked_identity_error"]) <= atol
        rows.append(row)
    return rows


def random_controls(model, probe, directory, event, full, seeds):
    rows = []
    for seed in seeds:
        unit = event["entry"]
        name = unit["unit"] + f"_random_{seed}"
        score = world(model, probe, directory, name, (unit_action(unit, operation="random", seed=seed),))
        rows.append(dict(event=event["event"], seed=seed, full=full, changed=score,
                         support=full - score, control="equal_norm_random_direction"))
    return rows


def audit_trials(model, probe, plan, directory, doses, random_seeds, atol):
    units = [unit for event in plan for unit in event["source_units"] if unit["sources"]]
    units += [event["relay"] for event in plan if event["relay"] is not None]
    probe = dict(probe, capture_units=units)
    record, writes = baseline(model, probe, directory)
    export_record(directory / "baseline.npz")
    full = record["measure"]
    effects, random = [], []
    for event in tqdm(plan, desc="native reanchor events", unit="event", leave=False):
        controls = event_controls(model, probe, directory, event, full, writes)
        for unit in tqdm(event["source_units"], desc="source messages", leave=False):
            if unit["sources"]:
                effects.extend(unit_trials(model, probe, directory, event, unit, doses,
                                            full, writes, controls, atol))
        random.extend(random_controls(model, probe, directory, event, full, random_seeds))
        pd.DataFrame(effects).to_csv(directory / "effects.csv", index=False)
        pd.DataFrame(random).to_csv(directory / "random_controls.csv", index=False)
    return record
