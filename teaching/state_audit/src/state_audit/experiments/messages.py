"""Head-resolved source messages composed from public capture and operation APIs."""

from dataclasses import dataclass
from itertools import groupby

import numpy as np
import torch

from ..capture import capture_targets, numpy
from ..intervention import intervene
from ..operations import Delete, Inject, Replace, Target
from .contrasts import score_contrast


@dataclass(frozen=True)
class MessageSite:
    """An explicit head group at one layer; keys=None selects its entire readout."""

    name: str
    layer: int
    heads: tuple[int, ...]
    positions: tuple[int, ...]
    keys: tuple[int, ...] | None = None

    @property
    def target(self):
        return Target("head_readout", (self.layer,), positions=self.positions, heads=self.heads)

    def deletion(self, strength=1.0):
        if self.keys is None:
            return Delete(self.target, strength)
        target = Target(
            "attention", (self.layer,), positions=self.positions, heads=self.heads, keys=self.keys
        )
        return Delete(target, strength)

    def observations(self, model):
        if self.keys is None:
            return {self.name: self.target}
        count, kv_count, _ = model.head_layout(self.layer)
        kv_heads = tuple(sorted({head // (count // kv_count) for head in self.heads}))
        return {
            self.name + ":total": self.target,
            self.name + ":attention": self.deletion().target,
            self.name + ":value": Target(
                "value", (self.layer,), positions=self.keys, heads=kv_heads
            ),
        }

    def read(self, model, captured):
        if self.keys is None:
            readout = captured[self.name][self.layer]
            return dict(readout=readout, total_readout=readout, mass=None)
        attention = captured[self.name + ":attention"][self.layer]
        values = captured[self.name + ":value"][self.layer]
        count, kv_count, _ = model.head_layout(self.layer)
        mapped = np.asarray(self.heads) // (count // kv_count)
        _, inverse = np.unique(mapped, return_inverse=True)
        readout = (attention @ values[inverse]).transpose(1, 0, 2)
        return dict(
            readout=readout,
            total_readout=captured[self.name + ":total"][self.layer],
            mass=attention.sum(-1).T,
        )

    def replacement(self, reference, current):
        """Replace a whole head or add donor-minus-current for only the selected sources."""
        if self.keys is None:
            return Replace(self.target, reference["readout"])
        return Inject(self.target, reference["readout"] - current["readout"])


def measure_messages(model, prefix_ids, candidates, sites, operations=()):
    """Read two continuations and capture selected messages in the causal prefix only."""
    targets = {}
    for site in sites:
        targets.update(site.observations(model))
    with torch.inference_mode(), intervene(model, operations):
        with capture_targets(model, targets) as captured:
            scores = score_contrast(model, prefix_ids, candidates)
        messages = {}
        for site in sites:
            values = site.read(model, captured)
            for name in ("readout", "total_readout"):
                write = model.project_heads(site.layer, values[name], site.heads)
                values[name.replace("readout", "write")] = numpy(write)
            messages[site.name] = values
    return scores, messages


def restore_messages(evaluate, sites, reference, operations=(), current=None):
    """Restore in layer order; recompute downstream messages after each earlier patch.

    evaluate returns (scores, messages). References are source-resolved head readouts.
    Within a layer, heads/queries are parallel; overlapping selections are disallowed
    by the experiment plan. Across layers a fresh current message is necessary.
    """
    changed = list(operations)
    if current is None:
        current = evaluate(changed)
    for _, layer_sites in groupby(sorted(sites, key=lambda site: site.layer), lambda s: s.layer):
        for site in layer_sites:
            changed.append(site.replacement(reference[site.name], current[1][site.name]))
        current = evaluate(changed)
    return current
