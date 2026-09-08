"""A narrow adapter for existing v3 captures; no new LLM forward pass."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np


class NativeCache:
    def __init__(self, path, weights):
        self.path, self.weights = Path(path), weights
        self.stack = ExitStack()
        with np.load(path, allow_pickle=False) as file:
            self.trace = dict(file)
        for name in ("states", "history", "qk"):
            setattr(self, name, self.stack.enter_context(np.load(self.path.with_suffix(f".{name}.npz"), allow_pickle=False)))
        self.layers, self.heads, self.rows = self.trace["head_margin"].shape
        cfg = weights.config
        if (self.layers, self.heads) != (cfg["num_hidden_layers"], cfg["num_attention_heads"]):
            self.stack.close()
            raise ValueError("capture and checkpoint disagree on physical layers/heads")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def source_partition(trace):
    """Every boundary/initial input belongs to exactly one named source.

    Material units are boundary positions, not unmixed semantic facts. Future
    response inputs have their own initial-state origin; numerical injections
    have a separate source and are never assigned to material.
    """
    n, start = len(trace["token_ids"]), int(trace["response_start"])
    special = trace["special_mask"]
    material = trace["evidence_mask"] & ~special & (np.arange(n) < start)
    units = trace["source_unit_id"]
    ids = sorted(set(units[material].tolist()))
    names = [f"material:{u}" for u in ids]
    kinds = ["material"] * len(ids)
    token_group = np.full(n, len(ids), int)
    for g, u in enumerate(ids):
        token_group[material & (units == u)] = g
    other, special_group, initial, rounding = range(len(ids), len(ids)+4)
    names += ["other_prompt", "special", "response_initial", "rounding"]
    kinds += ["other_prompt", "special", "response_initial", "rounding"]
    token_group[np.arange(n) >= start] = initial
    token_group[special] = special_group
    boundary = (np.arange(len(names))[:, None] == token_group[None]) & (np.arange(n) < start)[None]
    rows = trace["row_position"]
    initial_mask = np.arange(len(names))[:, None] == token_group[rows][None]
    return {"names": np.array(names), "kinds": np.array(kinds), "token_group": token_group,
            "boundary": boundary, "initial": initial_mask, "rounding": rounding}


def target_positions(trace, count=4):
    rows = trace["row_position"][:-1]
    valid = rows[~trace["special_mask"][rows] & ~trace["special_mask"][rows+1]] + 1
    if count and len(valid) > count:
        valid = valid[np.linspace(0, len(valid)-1, count).round().astype(int)]
    return valid.astype(int)
