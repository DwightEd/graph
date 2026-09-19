"""Four-world head interactions at one fixed prefix and candidate contrast."""

from itertools import combinations
import json

import numpy as np
import pandas as pd


PAIR_GROUPS = ("evidence", "past_history")


def select_pairs(writes, heads, pairs_per_group=2):
    """Freeze pair choices from the baseline, before any ablation is read.

    Preserve physical head identity. Prefer one aligned and one opposed pair
    per source group when available; this is discovery, not independent proof.
    """
    selected = set(heads)
    rows = writes[writes["head"].ge(0)]
    plans = []
    for group in PAIR_GROUPS:
        values = rows[rows.source_group == group]
        values = values.set_index(["layer", "head"]).local_linear_support
        candidates = []
        for left, right in combinations(sorted(selected), 2):
            a, b = float(values.loc[left]), float(values.loc[right])
            candidates.append(dict(source_group=group, left=list(left), right=list(right),
                                   local_left=a, local_right=b, opposed=a * b < 0,
                                   size=min(abs(a), abs(b))))
        ranked = sorted(candidates, key=lambda row: -row["size"])
        chosen = []
        for opposed in (False, True):
            match = next((row for row in ranked if row["opposed"] == opposed), None)
            if match is not None:
                chosen.append(match)
        chosen.extend(row for row in ranked if row not in chosen)
        plans.extend(chosen[:pairs_per_group])
    return plans


def interaction_values(full, without_left, without_right, without_both):
    """J = M(full) - M(-a) - M(-b) + M(-a,-b); units match M."""
    left = full - without_left
    right = full - without_right
    joint = full - without_both
    return dict(left_support=left, right_support=right, joint_support=joint,
                interaction=left + right - joint)


def run_pairs(model, probe, identity, baseline, plans, output):
    from .native import Intervention
    from .scoring import evaluate_candidates
    from .main import save_run

    stem = "_".join([identity["case_id"], identity["side"], identity["panel"]])
    directory = output / "coalitions"
    directory.mkdir(exist_ok=True)
    rows = []
    for plan in plans:
        left, right = plan["left"], plan["right"]
        group = plan["source_group"]
        name = f"L{left[0]}H{left[1]}_L{right[0]}H{right[1]}_{group}"
        actions = tuple(Intervention(layer, (group,), heads=(head,))
                        for layer, head in (left, right))
        path = directory / (stem + "_" + name + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                joint = json.loads(str(saved["record"]))
        else:
            joint, run = evaluate_candidates(model, probe, actions)
            save_run(path, dict(identity, **joint), run.trajectory, run.writes, run.changes)
        singles = [load_single(output, stem, group, head) for head in (left, right)]
        for metric in ("next_margin", "sequence_margin"):
            values = interaction_values(baseline[metric], singles[0][metric],
                                        singles[1][metric], joint[metric])
            rows.append(dict(identity, **plan, metric=metric, **values))
    return rows


def load_single(output, stem, group, coordinate):
    layer, head = coordinate
    path = output / "runs" / f"{stem}_L{layer}H{head}_{group}.npz"
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(str(saved["record"]))


def save_pairs(rows, output):
    columns = ["case_id", "side", "panel", "source_group", "left", "right",
               "metric", "left_support", "right_support", "joint_support", "interaction"]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    frame.to_csv(output / "head_interactions.csv", index=False)
    return frame
