"""A narrow adapter for existing v3 captures; no new LLM forward pass."""
from contextlib import ExitStack
from pathlib import Path
import re

import numpy as np

MASK_POLICY = 'capture_plus_control_markers_v1'
CONTROL = re.compile(r'<\|(?:begin_of_text|end_of_text|start_header_id|end_header_id|eot_id|eom_id|'
                     r'finetune_right_pad_id|python_tag|reserved_special_token_\d+)\|>')
META_FIELDS = ('token_ids','token_text','special_mask','source_unit_id','evidence_mask',
               'row_position','response_start','predictor_logprob','generator_model','settings',
               'capture_special_mask')


def repair_mask(trace):
    """Repair legacy classification, without deleting inputs or changing IDs.

    Match complete, known control-token spellings; ordinary punctuation and
    role words remain ordinary. This also works with moved, offline caches.
    """
    trace = dict(trace)
    native = np.asarray(trace.get('capture_special_mask',trace['special_mask']),bool)
    controls = np.array([bool(CONTROL.fullmatch(str(t))) for t in
                         trace.get('token_text',np.full(len(native),''))])
    trace['capture_special_mask'] = native.copy()
    trace['special_mask'] = np.asarray(trace['special_mask'],bool) | controls
    trace['added_special_positions'] = np.flatnonzero(trace['special_mask'] & ~native)
    return trace


def read_trace(path, *, runtime=False):
    """Read small fields only; never decompress the old exhaustive audit arrays."""
    fields = META_FIELDS + (('head_margin','readout_runner_id') if runtime else ())
    with np.load(path,allow_pickle=False) as data:
        trace = {k:data[k] for k in fields if k in data}
    return repair_mask(trace)


class NativeCache:
    def __init__(self, path, weights):
        self.path, self.weights = Path(path), weights
        self.stack = ExitStack()
        self.event_readouts = {}
        self.trace = read_trace(path, runtime=True)
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
        self.event_readouts.clear()
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
