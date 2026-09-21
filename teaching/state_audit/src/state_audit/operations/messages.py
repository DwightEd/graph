"""Replace selected source contributions inside the native attention contraction."""

from dataclasses import dataclass

import torch

from .target import Target


@dataclass
class ReplaceSource:
    """Reference weights [head, query, source] and values [head, source, width].

    Only target queries receive the replacement. Other source contributions remain
    current, including other queries sharing the same GQA value head.
    """

    target: Target
    weights: object
    values: object

    def insert(self, weights, values, position):
        target = self.target
        query_index = target.positions.index(position)
        donor_weights = torch.as_tensor(self.weights, device=weights.device, dtype=weights.dtype)
        donor_values = torch.as_tensor(self.values, device=values.device, dtype=values.dtype)
        count = donor_weights.shape[-1]
        keys = list(target.keys[:count])
        extra = max(0, count - len(keys))
        if extra:
            start = weights.shape[-1]
            keys.extend(range(start, start + extra))
            weights = torch.nn.functional.pad(weights, (0, extra))
            values = torch.nn.functional.pad(values, (0, 0, 0, extra))
        selected = Target("attention", target.layers, (position,), target.heads, target.keys)
        weights[selected.indices(weights)] = 0
        destination = Target("attention", target.layers, (position,), target.heads, tuple(keys))
        weights[destination.indices(weights)] = donor_weights[:, query_index : query_index + 1]
        destination = Target("value", target.layers, tuple(keys), target.heads)
        values[destination.indices(values)] = donor_values
        return weights, values


def replace_source_readouts(weights, values, replacements):
    """Contract full matrices in native dtype; never add separately rounded A_S V_S.

    Inputs have native [batch=1, head, query/source, feature/source] layouts.
    Reference sources occupy the selected slots; longer donors add explicit slots.
    A separate contraction per query keeps its donor values from leaking elsewhere.
    """
    output = weights @ values
    positions = sorted({q for operation in replacements for q in operation.target.positions})
    for position in positions:
        changed_weights, changed_values = weights[0].clone(), values[0].clone()
        for operation in replacements:
            if position in operation.target.positions:
                changed_weights, changed_values = operation.insert(
                    changed_weights, changed_values, position
                )
        restored = changed_weights.unsqueeze(0) @ changed_values.unsqueeze(0)
        output[:, :, position] = restored[:, :, position]
    return output
