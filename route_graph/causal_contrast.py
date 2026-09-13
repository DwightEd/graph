"""Finite continuation targets and typed message gates for graph audits.

These measurements are not factuality labels or a complete detector. Callers
must establish the semantic status of every continuation independently and
recompute the downstream network after each gate. No population code imports
this module.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContinuationContrast:
    """Disjoint token-sequence events after one common input prefix.

    ``continuations[0]`` is the observed event; subsequent events are the
    explicitly enumerated alternatives, not the universe of correct answers.
    The caller supplies complete, terminated statements without truncation.
    """

    prefix: tuple[int, ...]
    continuations: tuple[tuple[int, ...], ...]
    prompt_length: int

    @classmethod
    def from_sequences(cls, sequences, prompt_length):
        sequences = tuple(tuple(sequence) for sequence in sequences)
        if (
            len(sequences) < 2
            or type(prompt_length) is not int
            or prompt_length < 1
            or any(len(s) <= prompt_length for s in sequences)
            or any(type(t) is not int or t < 0 for s in sequences for t in s)
        ):
            raise ValueError("expected full token sequences and a nonempty prompt")
        # Sequence-prefix events overlap in an autoregressive sample space;
        # neither duplicate alternatives nor such events may be summed.
        for index, first in enumerate(sequences):
            for second in sequences[index + 1 :]:
                size = min(len(first), len(second))
                if first[:size] == second[:size]:
                    raise ValueError("continuation events overlap or are duplicates")
        common = 0
        for tokens in zip(*sequences):
            if len(set(tokens)) != 1:
                break
            common += 1
        if common < prompt_length:
            raise ValueError("contrast branches do not share the complete prompt")
        return cls(
            prefix=sequences[0][:common],
            continuations=tuple(s[common:] for s in sequences),
            prompt_length=prompt_length,
        )

    @property
    def shared_queries(self):
        """Include the query predicting the first divergent token, once."""
        return tuple(range(self.prompt_length - 1, len(self.prefix)))

    def score(self, token_logprobs):
        """Compute log P(observed) - log sum P(enumerated alternatives)."""
        probabilities = tuple(tuple(row) for row in token_logprobs)
        if len(probabilities) != len(self.continuations) or any(
            len(row) != len(tokens)
            for row, tokens in zip(probabilities, self.continuations)
        ):
            raise ValueError("every continuation token needs exactly one logp")
        if any(
            not math.isfinite(value) or value > 0
            for row in probabilities
            for value in row
        ):
            raise ValueError("token log probabilities must be finite and nonpositive")
        try:
            totals = tuple(math.fsum(row) for row in probabilities)
        except OverflowError as error:
            raise ValueError(
                "sequence log probability exceeds float64 range"
            ) from error
        maximum = max(totals[1:])
        # Subtract the common offset first, before adding a small log-sum term.
        # Otherwise even identical events at -1e308 lose the log(number) term.
        return (totals[0] - maximum) - math.log(
            math.fsum(math.exp(value - maximum) for value in totals[1:])
        )


def message_gate_delta(attention, values, selected, strength, kind, destination=None):
    """Return the typed gate's change before O as [query, head, head_dim].

    attention [head, query, role_key] is the CURRENT receiving computation;
    values [role_key, KV_head, head_dim] preserves GQA head identities. selected
    is one boolean mask over role keys. Queries/layers are selected by the
    caller. Causal invisibility must already be reflected by zero attention.

    ``content`` attenuates selected A*v messages. ``route`` redistributes
    within this role, preserving its current attention mass and all values.
    Neither operation modifies any other role. Invalid route normalization
    raises instead of turning an undefined intervention into a measurement.
    For a paired route transfer, destination is a disjoint boolean key mask:
    only this destination receives the mass removed from selected, in its
    current proportions. The caller specifies the source/history domain.
    """
    if kind not in {"route", "content"}:
        raise ValueError("kind must be route or content")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if (
        attention.ndim != 3
        or values.ndim != 3
        or any(size == 0 for size in (*attention.shape, *values.shape))
        or attention.shape[-1] != values.shape[0]
        or attention.shape[0] % values.shape[1]
        or attention.device != values.device
        or not attention.is_floating_point()
        or not values.is_floating_point()
    ):
        raise ValueError("incompatible role attention, values or GQA shapes")
    selected = torch.as_tensor(selected, device=attention.device)
    if selected.dtype != torch.bool or selected.shape != (values.shape[0],):
        raise ValueError("selected must be a boolean mask over role keys")
    if destination is not None:
        destination = torch.as_tensor(destination, device=attention.device)
        if (
            kind != "route"
            or destination.dtype != torch.bool
            or destination.shape != selected.shape
            or (destination & selected).any()
        ):
            raise ValueError("route destination must be a disjoint boolean key mask")
    if (
        not torch.isfinite(attention).all()
        or not torch.isfinite(values).all()
        or (attention < 0).any()
    ):
        raise ValueError("invalid attention or value entries")
    dtype = torch.promote_types(attention.dtype, values.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    weights, value = attention.to(dtype), values.to(dtype)
    original_mass = weights.sum(-1, keepdim=True)
    # An actual low-precision attention row may have rounded individual entries.
    # Its storage error is different from a universal 1% probability allowance.
    mass_tolerance = torch.finfo(attention.dtype).eps / 2 + 1e-6
    if (original_mass > 1 + mass_tolerance).any():
        raise ValueError("role attention exceeds total attention mass")
    shape = (attention.shape[1], attention.shape[0], values.shape[-1])
    if strength == 1 or not selected.any():
        return torch.zeros(shape, device=values.device, dtype=dtype), {
            "mass_error": 0.0
        }
    factor = torch.ones(values.shape[0], device=values.device, dtype=dtype)
    factor[selected] = strength
    if kind == "content":
        change = weights * (factor - 1)
        mass_error = 0.0  # The actual routing weights are unchanged.
    elif destination is None:
        changed_weights = weights * factor
        remaining = changed_weights.sum(-1, keepdim=True)
        if ((original_mass > 0) & (remaining == 0)).any():
            raise ValueError("route gate leaves no key for nonzero role mass")
        safe_mass = torch.where(remaining > 0, remaining, 1.0)
        changed_weights = changed_weights * (original_mass / safe_mass)
        mass_error = float(
            (changed_weights.sum(-1, keepdim=True) - original_mass).abs().max()
        )
        change = changed_weights - weights
    else:
        removed = weights * (1 - factor)
        incoming = removed.sum(-1, keepdim=True)
        receiving = weights * destination
        receiving_mass = receiving.sum(-1, keepdim=True)
        if ((incoming > 0) & (receiving_mass == 0)).any():
            raise ValueError("route destination has no receiving mass")
        safe_mass = torch.where(receiving_mass > 0, receiving_mass, 1.0)
        change = receiving / safe_mass * incoming - removed
        mass_error = float(change.sum(-1).abs().max())
    repeats = attention.shape[0] // values.shape[1]
    grouped = change.reshape(values.shape[1], repeats, attention.shape[1], -1)
    delta = torch.einsum("grqk,kgd->qgrd", grouped, value).reshape(shape)
    if not torch.isfinite(delta).all():
        raise ValueError("message delta exceeds working dtype range")
    return delta, {"mass_error": mass_error}
