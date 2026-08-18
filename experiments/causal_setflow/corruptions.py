"""Invariant-aware causal source-set corruptions for MG-CASF.

Corruptions are synthetic self-supervision only.  They never read hallucination
labels and operate after exact RR source sets have been materialized.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import CORRUPTION_NAMES, CorruptionConfig
from .data import LayerSourceSets


@dataclass(frozen=True)
class CorruptionPlan:
    """One contiguous token/layer corruption with a selected head subset."""

    type_index: int
    token_mask: torch.Tensor
    layer_mask: torch.Tensor
    head_mask: torch.Tensor

    @property
    def name(self) -> str:
        return CORRUPTION_NAMES[int(self.type_index)]

    def validate(
        self, *, response_count: int, num_layers: int, num_heads: int
    ) -> "CorruptionPlan":
        if not 0 <= int(self.type_index) < len(CORRUPTION_NAMES):
            raise ValueError("corruption type index is invalid")
        expected = (
            (self.token_mask, response_count, "token"),
            (self.layer_mask, num_layers, "layer"),
            (self.head_mask, num_heads, "head"),
        )
        for value, length, name in expected:
            if value.dtype != torch.bool or value.shape != (int(length),):
                raise ValueError(f"corruption {name} mask has invalid geometry")
            if not bool(value.any()):
                raise ValueError(f"corruption {name} mask is empty")
        if bool(self.token_mask[0]):
            raise ValueError("corruption must not select the first response token")
        return self


def sample_corruption_plan(
    response_count: int,
    num_layers: int,
    num_heads: int,
    config: CorruptionConfig,
    *,
    device: str | torch.device,
    generator: torch.Generator,
    forced_type: int | None = None,
) -> CorruptionPlan:
    """Sample one balanced, online-causal structural corruption plan."""

    config.validate()
    response_count = int(response_count)
    num_layers = int(num_layers)
    num_heads = int(num_heads)
    if response_count < 2 or num_layers < 1 or num_heads < 1:
        raise ValueError("corruption sampling needs at least two response tokens")
    device = torch.device(device)
    type_index = (
        int(forced_type) % len(CORRUPTION_NAMES)
        if forced_type is not None
        else int(
            torch.randint(
                len(CORRUPTION_NAMES),
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
    )

    available_tokens = response_count - 1
    token_length = _sample_length(
        config.token_span_min,
        config.token_span_max,
        available_tokens,
        generator=generator,
        device=device,
    )
    token_start = 1 + _randint(
        max(1, available_tokens - token_length + 1),
        generator=generator,
        device=device,
    )
    token_mask = torch.zeros(response_count, dtype=torch.bool, device=device)
    token_mask[token_start : token_start + token_length] = True

    layer_length = _sample_length(
        config.layer_span_min,
        config.layer_span_max,
        num_layers,
        generator=generator,
        device=device,
    )
    layer_start = _randint(
        max(1, num_layers - layer_length + 1),
        generator=generator,
        device=device,
    )
    layer_mask = torch.zeros(num_layers, dtype=torch.bool, device=device)
    layer_mask[layer_start : layer_start + layer_length] = True

    head_count = max(1, round(num_heads * float(config.selected_head_fraction)))
    if CORRUPTION_NAMES[type_index] == "homogenize" and num_heads > 1:
        head_count = max(2, head_count)
    head_count = min(num_heads, head_count)
    order = torch.randperm(num_heads, generator=generator, device=device)
    head_mask = torch.zeros(num_heads, dtype=torch.bool, device=device)
    head_mask[order[:head_count]] = True
    return CorruptionPlan(
        type_index=type_index,
        token_mask=token_mask,
        layer_mask=layer_mask,
        head_mask=head_mask,
    ).validate(
        response_count=response_count,
        num_layers=num_layers,
        num_heads=num_heads,
    )


@torch.no_grad()
def apply_corruption(
    source_sets: LayerSourceSets,
    plan: CorruptionPlan,
    *,
    layer_index: int,
    config: CorruptionConfig,
) -> tuple[LayerSourceSets, torch.Tensor]:
    """Apply one corruption and return the exact changed `(token, head)` mask."""

    config.validate()
    tokens, heads = source_sets.total_mass.shape
    plan.validate(
        response_count=tokens,
        num_layers=len(plan.layer_mask),
        num_heads=heads,
    )
    if not bool(plan.layer_mask[int(layer_index)]):
        return source_sets, torch.zeros(
            (tokens, heads), dtype=torch.bool, device=source_sets.total_mass.device
        )

    values = {
        name: getattr(source_sets, name).clone()
        for name in source_sets.__dataclass_fields__
    }
    active = values["route_mask"].any(dim=-1) | values["memory_mask"].any(dim=-1)
    selected = (
        plan.token_mask[:, None]
        & plan.head_mask[None, :]
        & active
    )
    if not bool(selected.any()):
        return source_sets, selected

    name = plan.name
    if name == "collapse":
        changed = _collapse(values, selected, config)
    elif name == "localize":
        changed = _localize(values, selected)
    elif name == "freeze":
        changed = _freeze(values, selected, config)
    elif name == "homogenize":
        changed = _homogenize(values, selected, config)
    elif name == "self_reinforce":
        changed = _self_reinforce(values, selected, config)
    else:  # pragma: no cover - protected by plan validation
        raise KeyError(name)

    corrupted = replace(source_sets, **values).validate()
    return corrupted, changed


def _collapse(values, selected, config):
    route_old = values["route_weight"]
    route_new = _power_reweight(
        route_old,
        values["route_mask"],
        float(config.collapse_power),
        config.epsilon,
    )
    memory_old = values["memory_received"]
    memory_new = _power_reweight(
        memory_old,
        values["memory_mask"],
        float(config.collapse_power),
        config.epsilon,
    )
    mask = selected[..., None]
    values["route_weight"] = torch.where(mask, route_new, route_old)
    values["route_received_delta"] = torch.where(
        mask,
        values["route_received_delta"] + route_new - route_old,
        values["route_received_delta"],
    )
    values["memory_received"] = torch.where(mask, memory_new, memory_old)
    values["memory_received_delta"] = torch.where(
        mask,
        values["memory_received_delta"] + memory_new - memory_old,
        values["memory_received_delta"],
    )
    return selected


def _self_reinforce(values, selected, config):
    route_old = values["route_weight"]
    route_new = _factor_reweight(
        route_old,
        values["route_received"],
        values["route_mask"],
        float(config.self_reinforce_power),
        config.epsilon,
    )
    memory_old = values["memory_current_weight"]
    memory_new = _factor_reweight(
        memory_old,
        values["memory_received"],
        values["memory_mask"],
        float(config.self_reinforce_power),
        config.epsilon,
    )
    mask = selected[..., None]
    values["route_weight"] = torch.where(mask, route_new, route_old)
    values["memory_current_weight"] = torch.where(mask, memory_new, memory_old)
    return selected


def _localize(values, selected):
    changed = torch.zeros_like(selected)
    for token, head in torch.nonzero(selected, as_tuple=False).tolist():
        if token < 1:
            continue
        for prefix in ("route", "memory"):
            valid = values[f"{prefix}_mask"][token, head]
            count = int(valid.sum().item())
            if count < 1:
                continue
            recent = torch.arange(
                token - 1,
                max(-1, token - count - 1),
                -1,
                device=valid.device,
                dtype=values[f"{prefix}_source"].dtype,
            )
            values[f"{prefix}_source"][token, head, :count] = recent[:count]
            changed[token, head] = True
    return changed


def _freeze(values, selected, config):
    changed = torch.zeros_like(selected)
    eps = float(config.epsilon)
    for token, head in torch.nonzero(selected, as_tuple=False).tolist():
        if token < 1:
            continue
        for prefix in ("route", "memory"):
            valid_name = f"{prefix}_mask"
            previous_valid = values[valid_name][token - 1, head]
            if not bool(previous_valid.any()):
                continue
            current_valid = values[valid_name][token, head]
            values[valid_name][token, head] = previous_valid
            values[f"{prefix}_source"][token, head] = values[
                f"{prefix}_source"
            ][token - 1, head]
            if prefix == "route":
                old_sum = values["route_weight"][token, head][current_valid].sum()
                previous = values["route_weight"][token - 1, head]
                values["route_weight"][token, head] = _rescale_vector(
                    previous, previous_valid, old_sum, eps
                )
                values["route_received"][token, head] = values[
                    "route_received"
                ][token - 1, head]
                values["route_received_delta"][token, head].zero_()
            else:
                old_received = values["memory_received"][token, head][
                    current_valid
                ].sum()
                old_current = values["memory_current_weight"][token, head][
                    current_valid
                ].sum()
                values["memory_received"][token, head] = _rescale_vector(
                    values["memory_received"][token - 1, head],
                    previous_valid,
                    old_received,
                    eps,
                )
                values["memory_current_weight"][token, head] = _rescale_vector(
                    values["memory_current_weight"][token - 1, head],
                    previous_valid,
                    old_current,
                    eps,
                )
                values["memory_received_delta"][token, head].zero_()
            changed[token, head] = True
    return changed


def _homogenize(values, selected, config):
    changed = torch.zeros_like(selected)
    eps = float(config.epsilon)
    for token in torch.nonzero(selected.any(dim=1), as_tuple=False).flatten().tolist():
        heads = torch.nonzero(selected[token], as_tuple=False).flatten()
        if len(heads) < 2:
            continue
        mass = values["total_mass"][token, heads]
        anchor = int(heads[int(torch.argmax(mass).item())].item())
        for head_tensor in heads:
            head = int(head_tensor.item())
            if head == anchor:
                continue
            for prefix in ("route", "memory"):
                mask_name = f"{prefix}_mask"
                anchor_valid = values[mask_name][token, anchor]
                if not bool(anchor_valid.any()):
                    continue
                current_valid = values[mask_name][token, head]
                values[mask_name][token, head] = anchor_valid
                values[f"{prefix}_source"][token, head] = values[
                    f"{prefix}_source"
                ][token, anchor]
                if prefix == "route":
                    target = values["route_weight"][token, head][current_valid].sum()
                    values["route_weight"][token, head] = _rescale_vector(
                        values["route_weight"][token, anchor],
                        anchor_valid,
                        target,
                        eps,
                    )
                    values["route_received"][token, head] = values[
                        "route_received"
                    ][token, anchor]
                    values["route_received_delta"][token, head] = values[
                        "route_received_delta"
                    ][token, anchor]
                else:
                    target_received = values["memory_received"][token, head][
                        current_valid
                    ].sum()
                    target_current = values["memory_current_weight"][token, head][
                        current_valid
                    ].sum()
                    values["memory_received"][token, head] = _rescale_vector(
                        values["memory_received"][token, anchor],
                        anchor_valid,
                        target_received,
                        eps,
                    )
                    values["memory_current_weight"][token, head] = _rescale_vector(
                        values["memory_current_weight"][token, anchor],
                        anchor_valid,
                        target_current,
                        eps,
                    )
                    values["memory_received_delta"][token, head] = values[
                        "memory_received_delta"
                    ][token, anchor]
                changed[token, head] = True
    return changed


def _power_reweight(values, valid, power, epsilon):
    positive = torch.where(valid, values.clamp_min(0.0), torch.zeros_like(values))
    target = positive.sum(dim=-1, keepdim=True)
    powered = positive.pow(float(power))
    denominator = powered.sum(dim=-1, keepdim=True)
    result = powered / denominator.clamp_min(float(epsilon)) * target
    return torch.where(valid, result, torch.zeros_like(result))


def _factor_reweight(values, factor, valid, power, epsilon):
    positive = torch.where(valid, values.clamp_min(0.0), torch.zeros_like(values))
    target = positive.sum(dim=-1, keepdim=True)
    multiplier = torch.where(
        valid,
        factor.clamp_min(float(epsilon)).pow(float(power)),
        torch.zeros_like(factor),
    )
    weighted = positive * multiplier
    result = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(
        float(epsilon)
    ) * target
    return torch.where(valid, result, torch.zeros_like(result))


def _rescale_vector(values, valid, target_sum, epsilon):
    result = torch.where(valid, values.clamp_min(0.0), torch.zeros_like(values))
    total = result.sum()
    if float(total) <= float(epsilon):
        count = valid.sum().clamp_min(1)
        return torch.where(
            valid,
            torch.as_tensor(target_sum, device=result.device, dtype=result.dtype)
            / count.to(result.dtype),
            torch.zeros_like(result),
        )
    return result / total * torch.as_tensor(
        target_sum, device=result.device, dtype=result.dtype
    )


def _sample_length(minimum, maximum, available, *, generator, device):
    low = min(int(minimum), int(available))
    high = min(int(maximum), int(available))
    if low > high:
        low = high
    if low == high:
        return low
    return low + _randint(high - low + 1, generator=generator, device=device)


def _randint(high, *, generator, device):
    if int(high) <= 1:
        return 0
    return int(
        torch.randint(
            int(high), (1,), generator=generator, device=device
        ).item()
    )