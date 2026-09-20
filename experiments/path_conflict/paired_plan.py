"""Use the same physical heads on both sides and at every observed phase."""

from itertools import combinations

import numpy as np
import pandas as pd

from .paired_inputs import SOURCE_GROUPS


def target_head_scores(arrays):
    """Rank the receiver actually intervened on, not an earlier head occurrence."""
    support = np.abs(arrays["target_sources"]).sum(axis=-1)
    return pd.DataFrame([
        dict(layer=layer, head=head, final_linear_support=float(support[layer, head]))
        for layer in range(support.shape[0]) for head in range(support.shape[1])
    ])


def choose_paired_heads(edge_tables, layers, heads, count, controls, seed):
    scores = []
    for table in edge_tables:
        scores.append(table.assign(strength=table.final_linear_support.abs()).groupby(
            ["layer", "head"], as_index=False).strength.max())
    combined = pd.concat(scores).groupby(["layer", "head"], as_index=False).strength.max()
    ranked = combined.sort_values(["strength", "layer", "head"], ascending=[False, True, True])
    selected = [dict(layer=int(row.layer), head=int(row.head), selection="paired_gradient")
                for row in ranked.head(count).itertuples()]
    used = {(row["layer"], row["head"]) for row in selected}
    # Controls come from the actual remaining model heads, including those not
    # present in the sparse gradient table. They are not 'unselected top edges'.
    pool = [(layer, head) for layer in range(layers) for head in range(heads)
            if (layer, head) not in used]
    rng = np.random.default_rng(seed)
    for index in rng.choice(len(pool), min(controls, len(pool)), replace=False):
        layer, head = pool[index]
        selected.append(dict(layer=layer, head=head, selection="model_head_control"))
    return selected


def head_units(heads, probe):
    units = []
    for head in heads:
        for group in SOURCE_GROUPS:
            units.append(dict(head, unit=f"L{head['layer']}H{head['head']}_{group}",
                receiver=len(probe["prefix_ids"]) - 1, source_group=group,
                sources=list(map(int, probe["groups"][group]))))
    return units


def joint_plan(heads):
    """All selected-head pairs, plus disjoint evidence/history within each head.

    Pairing is structural and frozen before finite effects are read. An edge
    sign is never used to call a pair cooperative or antagonistic.
    """
    names = [f"L{row['layer']}H{row['head']}" for row in heads
             if row["selection"] == "paired_gradient"]
    pairs = [dict(left=left + "_head_total", right=right + "_head_total",
                  relation="different_heads") for left, right in combinations(names, 2)]
    pairs.extend(dict(left=name + "_evidence", right=name + "_history",
                      relation="evidence_history") for name in names)
    pairs.extend(dict(left=name + "_evidence", right=name + "_inapplicable_source",
                      relation="applicable_inapplicable") for name in names)
    return pairs
