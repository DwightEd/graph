"""Freeze signed message candidates and head pairs before finite interventions."""

from itertools import combinations

import numpy as np


def select_units(edges, max_units=8, control_units=2, seed=0):
    """Balance both signs and retain physical heads; source roles are never read."""
    ordered = edges.assign(strength=edges.final_linear_support.abs()).sort_values(
        ["strength", "unit"], ascending=[False, True], kind="stable")
    positive = ordered[ordered.final_linear_support > 0].to_dict("records")
    negative = ordered[ordered.final_linear_support < 0].to_dict("records")
    selected, used_heads = [], set()
    for rank in range(max(len(positive), len(negative))):
        for rows in (positive, negative):
            if rank < len(rows) and len(selected) < max_units:
                row = rows[rank]
                head = (row["layer"], row["head"])
                if head not in used_heads:
                    selected.append(dict(row, selection="signed_gradient"))
                    used_heads.add(head)
    used = {row["unit"] for row in selected}
    remaining = ordered[~ordered.unit.isin(used)].to_dict("records")
    rng = np.random.default_rng(seed)
    count = min(control_units, len(remaining))
    for index in rng.choice(len(remaining), size=count, replace=False):
        selected.append(dict(remaining[index], selection="unselected_candidate_control"))
    return selected


def pair_kind(left, right):
    a, b = left["final_linear_support"], right["final_linear_support"]
    if a * b < 0:
        return "opposed"
    return "co_support" if a > 0 else "co_suppress"


def select_message_pairs(units, max_pairs=4):
    units = [row for row in units if row["selection"] == "signed_gradient"]
    candidates = []
    for left, right in combinations(units, 2):
        strength = min(abs(left["final_linear_support"]), abs(right["final_linear_support"]))
        candidates.append(dict(left=left["unit"], right=right["unit"],
                               kind=pair_kind(left, right), strength=strength))
    ranked = sorted(candidates, key=lambda row: (-row["strength"], row["left"], row["right"]))
    chosen = []
    for kind in ("opposed", "co_support", "co_suppress"):
        match = next((row for row in ranked if row["kind"] == kind), None)
        if match is not None:
            chosen.append(match)
    chosen.extend(row for row in ranked if row not in chosen)
    return chosen[:max_pairs]


def causal_order(left, right):
    """A receiver can only affect a later layer at the same or later position."""
    early, late = sorted((left, right), key=lambda row: row["layer"])
    if early["layer"] < late["layer"] and early["receiver"] <= late["receiver"]:
        return early, late
    return None
