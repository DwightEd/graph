"""Repeatable masks for source members, heads, and Transformer layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerMaskPlan:
    route_element: torch.Tensor
    memory_element: torch.Tensor
    head: torch.Tensor


@dataclass(frozen=True)
class SequenceMaskPlan:
    layer: torch.Tensor


def bernoulli_valid_mask(
    valid: torch.Tensor,
    probability: float,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    valid = valid.bool()
    if float(probability) <= 0.0:
        return torch.zeros_like(valid)
    random = torch.rand(valid.shape, generator=generator, device=valid.device)
    masked = valid & (random < float(probability))
    rows = valid.reshape(-1, valid.shape[-1])
    output = masked.reshape_as(rows)
    missing = rows.any(dim=1) & ~output.any(dim=1)
    if bool(missing.any()):
        first = rows.to(torch.int64).argmax(dim=1)
        selected = torch.nonzero(missing, as_tuple=False).squeeze(1)
        output[selected, first[selected]] = True
    return output.reshape_as(valid)


def sample_layer_mask_plan(
    route_valid: torch.Tensor,
    memory_valid: torch.Tensor,
    *,
    element_probability: float,
    head_probability: float,
    generator: torch.Generator | None,
) -> LayerMaskPlan:
    route = bernoulli_valid_mask(
        route_valid, element_probability, generator=generator
    )
    memory = bernoulli_valid_mask(
        memory_valid, element_probability, generator=generator
    )
    active = route_valid.any(dim=-1) | memory_valid.any(dim=-1)
    random = torch.rand(active.shape, generator=generator, device=active.device)
    head = active & (random < float(head_probability))
    missing = active.any(dim=1) & ~head.any(dim=1)
    if float(head_probability) > 0.0 and bool(missing.any()):
        first = active.to(torch.int64).argmax(dim=1)
        selected = torch.nonzero(missing, as_tuple=False).squeeze(1)
        head[selected, first[selected]] = True
    # Whole-head reconstruction uses a clean set-state target. Element masks
    # are therefore disabled for rows whose entire head will be hidden later.
    route = route & ~head.unsqueeze(-1)
    memory = memory & ~head.unsqueeze(-1)
    return LayerMaskPlan(route_element=route, memory_element=memory, head=head)


def sample_sequence_mask_plan(
    num_layers: int,
    probability: float,
    *,
    device: str | torch.device,
    generator: torch.Generator | None,
) -> SequenceMaskPlan:
    random = torch.rand((int(num_layers),), generator=generator, device=device)
    layer = random < float(probability)
    if int(num_layers) > 1 and float(probability) > 0.0 and not bool(layer.any()):
        layer[int(torch.argmin(random).item())] = True
    return SequenceMaskPlan(layer=layer)


def deterministic_generator(seed: int, *, device: str | torch.device) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device).type)
    generator.manual_seed(int(seed))
    return generator
