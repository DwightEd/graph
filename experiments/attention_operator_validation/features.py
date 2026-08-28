"""Answer-level mechanism features from pair-specific attention operator codes."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from experiments.grounded_route.graph import TokenGraph

from .operators import OperatorGeometry
from .pair_codes import PAIR_RETAINED, PAIR_SELF, PairCodeField


OPERATOR_MODES = (
    "identity",
    "operator_raw",
    "operator_normalized",
    "operator_permuted",
)


def _finite_mean(value: torch.Tensor) -> float:
    value = value.flatten().float()
    value = value[torch.isfinite(value)]
    return float(value.mean().item()) if len(value) else float("nan")


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask]
    return _finite_mean(selected)


def _response_masks(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(1, math.ceil(count / 3))
    position = torch.arange(count)
    return position < width, position >= max(count - width, 0)


def _layer_masks(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(1, math.ceil(count / 3))
    position = torch.arange(count)
    return position < width, position >= max(count - width, 0)


def _mean_slope(value: torch.Tensor) -> float:
    """Mean response-position slope across layers."""

    if value.ndim != 2 or value.shape[1] < 2:
        return 0.0
    x = torch.linspace(-1.0, 1.0, value.shape[1])
    denominator = x.square().sum().clamp_min(1e-12)
    centered = value - value.mean(dim=1, keepdim=True)
    slope = (centered * x).sum(dim=1) / denominator
    return _finite_mean(slope)


def _weighted_effective_heads(
    code: torch.Tensor,
    probability: torch.Tensor,
) -> torch.Tensor:
    """Average ``exp(H(head-code))`` over observed role/source pairs."""

    if code.ndim != 2 or probability.shape != (code.shape[0],):
        raise ValueError("head codes and pair probabilities are misaligned")
    entropy = -(code * code.clamp_min(1e-12).log()).sum(dim=1)
    return (probability * entropy.exp()).sum()


def _role_pair_statistics(
    graph: TokenGraph,
    field: PairCodeField,
) -> dict[str, torch.Tensor]:
    """Compute source breadth, depth, head entropy and observation coverage."""

    layers, responses = graph.layer_count, graph.response_count
    shape = (layers, responses)
    prompt_effective = torch.zeros(shape)
    history_effective = torch.zeros(shape)
    total_effective = torch.zeros(shape)
    prompt_top1 = torch.zeros(shape)
    history_top1 = torch.zeros(shape)
    total_top1 = torch.zeros(shape)
    prompt_lag = torch.zeros(shape)
    history_lag = torch.zeros(shape)
    total_lag = torch.zeros(shape)
    # Zero is outside the semantic range of an effective head count.  Missing
    # role/source pairs therefore remain unavailable NaN and are summarized
    # conditionally; their coverage is reported separately below.
    prompt_head_effective = torch.full(shape, float("nan"))
    history_head_effective = torch.full(shape, float("nan"))
    prompt_observed = torch.zeros(shape)
    history_observed = torch.zeros(shape)

    retained = field.kind == PAIR_RETAINED
    row = field.layer * responses + (field.target - graph.response_start)
    prompt = retained & (field.source < graph.response_start)
    history = retained & (field.source >= graph.response_start)

    def fill(mask: torch.Tensor, effective, top1, lag, head_effective, observed):
        selected_groups = torch.unique(row[mask])
        for group in selected_groups.tolist():
            current = mask & (row == int(group)) & (field.magnitude > 0)
            weight = field.magnitude[current]
            if not len(weight) or float(weight.sum().item()) <= 0:
                continue
            probability = weight / weight.sum()
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
            layer = int(group) // responses
            response = int(group) % responses
            effective[layer, response] = entropy.exp()
            top1[layer, response] = probability.max()
            source = field.source[current].float()
            target = field.target[current].float().clamp_min(1.0)
            lag[layer, response] = (
                probability * ((target - source) / target)
            ).sum()
            code = field.direction[current]
            head_effective[layer, response] = _weighted_effective_heads(
                code,
                probability,
            )
            coverage = field.observed[current].float().mean(dim=1)
            observed[layer, response] = (probability * coverage).sum()

    fill(
        prompt,
        prompt_effective,
        prompt_top1,
        prompt_lag,
        prompt_head_effective,
        prompt_observed,
    )
    fill(
        history,
        history_effective,
        history_top1,
        history_lag,
        history_head_effective,
        history_observed,
    )

    for group in torch.unique(row[retained]).tolist():
        current = retained & (row == int(group)) & (field.magnitude > 0)
        weight = field.magnitude[current]
        if not len(weight) or float(weight.sum().item()) <= 0:
            continue
        probability = weight / weight.sum()
        layer = int(group) // responses
        response = int(group) % responses
        total_effective[layer, response] = (
            -(probability * probability.clamp_min(1e-12).log()).sum()
        ).exp()
        total_top1[layer, response] = probability.max()
        source = field.source[current].float()
        target = field.target[current].float().clamp_min(1.0)
        total_lag[layer, response] = (
            probability * ((target - source) / target)
        ).sum()

    return {
        "prompt_effective": prompt_effective,
        "history_effective": history_effective,
        "total_effective": total_effective,
        "prompt_top1": prompt_top1,
        "history_top1": history_top1,
        "total_top1": total_top1,
        "prompt_lag": prompt_lag,
        "history_lag": history_lag,
        "total_lag": total_lag,
        "prompt_head_effective": prompt_head_effective,
        "history_head_effective": history_head_effective,
        "prompt_observed": prompt_observed,
        "history_observed": history_observed,
    }


def _role_mass(
    graph: TokenGraph,
    field: PairCodeField,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return retained prompt/history/self mass and native unresolved mass."""

    rows = graph.layer_count * graph.response_count
    mass = torch.zeros((rows, 3), dtype=torch.float32)
    row = field.layer * graph.response_count + (
        field.target - graph.response_start
    )
    retained = field.kind == PAIR_RETAINED
    prompt = retained & (field.source < graph.response_start)
    history = retained & (field.source >= graph.response_start)
    self_pair = field.kind == PAIR_SELF
    if bool(prompt.any()):
        mass[:, 0].index_add_(0, row[prompt], field.magnitude[prompt])
    if bool(history.any()):
        mass[:, 1].index_add_(0, row[history], field.magnitude[history])
    if bool(self_pair.any()):
        mass[:, 2].index_add_(0, row[self_pair], field.magnitude[self_pair])
    mass = mass.view(graph.layer_count, graph.response_count, 3)
    unresolved = graph.unresolved.permute(1, 0, 2).mean(dim=-1).cpu().float()
    return mass, unresolved


def _operator_role_state(
    graph: TokenGraph,
    field: PairCodeField,
    geometry: OperatorGeometry,
    mode: str,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Compute pair-code dispersion and same-layer generation dynamics."""

    layers, responses, heads = (
        graph.layer_count,
        graph.response_count,
        graph.head_count,
    )
    embedding = field.operator_embedding(
        geometry,
        mode,
        seed=seed,
        use_direction=True,
    )
    retained = field.kind == PAIR_RETAINED
    prompt = retained & (field.source < graph.response_start)
    history = retained & (field.source >= graph.response_start)
    row = field.layer * responses + (field.target - graph.response_start)

    role_mass = torch.zeros((layers * responses, 2))
    role_sum = torch.zeros((layers * responses, 2, heads))
    role_second = torch.zeros((layers * responses, 2))
    for role, mask in enumerate((prompt, history)):
        if not bool(mask.any()):
            continue
        group = row[mask]
        weight = field.magnitude[mask]
        role_mass[:, role].index_add_(0, group, weight)
        role_sum[:, role].index_add_(0, group, embedding[mask] * weight[:, None])
        role_second[:, role].index_add_(
            0,
            group,
            embedding[mask].square().sum(dim=1) * weight,
        )
    denominator = role_mass.clamp_min(1e-12)
    mean = role_sum / denominator[..., None]
    dispersion = (
        role_second / denominator - mean.square().sum(dim=-1)
    ).clamp_min(0.0)
    valid = role_mass > 1e-12
    mean = mean.view(layers, responses, 2, heads)
    dispersion = dispersion.view(layers, responses, 2)
    valid = valid.view(layers, responses, 2)

    prompt_mean = mean[:, :, 0]
    history_mean = mean[:, :, 1]
    prompt_history_distance = (
        prompt_mean - history_mean
    ).square().sum(dim=-1)
    prompt_history_normalizer = (
        prompt_mean.square().sum(dim=-1)
        + history_mean.square().sum(dim=-1)
    ).clamp_min(1e-12)
    prompt_history_distance = prompt_history_distance / prompt_history_normalizer
    prompt_history_valid = valid[:, :, 0] & valid[:, :, 1]

    def step_switch(role: int) -> tuple[torch.Tensor, torch.Tensor]:
        left = mean[:, :-1, role]
        right = mean[:, 1:, role]
        distance = (right - left).square().sum(dim=-1)
        normalizer = (
            right.square().sum(dim=-1) + left.square().sum(dim=-1)
        ).clamp_min(1e-12)
        distance = distance / normalizer
        pair_valid = valid[:, :-1, role] & valid[:, 1:, role]
        return distance, pair_valid

    prompt_switch, prompt_switch_valid = step_switch(0)
    history_switch, history_switch_valid = step_switch(1)

    self_pair = field.kind == PAIR_SELF
    self_embedding = torch.zeros((layers * responses, heads))
    self_valid = torch.zeros(layers * responses, dtype=torch.bool)
    if bool(self_pair.any()):
        self_row = row[self_pair]
        self_embedding[self_row] = embedding[self_pair]
        self_valid[self_row] = field.magnitude[self_pair] > 1e-12
    self_embedding = self_embedding.view(layers, responses, heads)
    self_valid = self_valid.view(layers, responses)
    self_switch = (
        self_embedding[:, 1:] - self_embedding[:, :-1]
    ).square().sum(dim=-1)
    self_normalizer = (
        self_embedding[:, 1:].square().sum(dim=-1)
        + self_embedding[:, :-1].square().sum(dim=-1)
    ).clamp_min(1e-12)
    self_switch = self_switch / self_normalizer
    self_switch_valid = self_valid[:, 1:] & self_valid[:, :-1]

    return {
        "dispersion": dispersion,
        "valid": valid,
        "prompt_history_distance": prompt_history_distance,
        "prompt_history_valid": prompt_history_valid,
        "prompt_switch": prompt_switch,
        "prompt_switch_valid": prompt_switch_valid,
        "history_switch": history_switch,
        "history_switch_valid": history_switch_valid,
        "self_switch": self_switch,
        "self_switch_valid": self_switch_valid,
    }


def _selected_mean(value: torch.Tensor, valid: torch.Tensor) -> float:
    return _finite_mean(value[valid])


def extract_answer_features(
    graph: TokenGraph,
    field: PairCodeField,
    geometry: OperatorGeometry,
    *,
    seed: int = 20260828,
    modes: Iterable[str] = OPERATOR_MODES,
) -> dict[str, float]:
    """Extract label-free answer-level routing and operator-dynamics features."""

    if geometry.layer_count != graph.layer_count or geometry.head_count != graph.head_count:
        raise ValueError("operator geometry and attention graph have different layer/head counts")
    early_response, late_response = _response_masks(graph.response_count)
    early_layer, late_layer = _layer_masks(graph.layer_count)
    role_mass, unresolved = _role_mass(graph, field)
    prompt_mass = role_mass[:, :, 0]
    history_mass = role_mass[:, :, 1]
    self_mass = role_mass[:, :, 2]
    retained_mass = role_mass.sum(dim=-1)
    row_coverage = retained_mass + unresolved

    feature: dict[str, float] = {
        "prompt_mass_mean": _finite_mean(prompt_mass),
        "prompt_mass_early": _masked_mean(prompt_mass, early_response[None].expand_as(prompt_mass)),
        "prompt_mass_late": _masked_mean(prompt_mass, late_response[None].expand_as(prompt_mass)),
        "prompt_mass_slope": _mean_slope(prompt_mass),
        "history_mass_mean": _finite_mean(history_mass),
        "history_mass_early": _masked_mean(history_mass, early_response[None].expand_as(history_mass)),
        "history_mass_late": _masked_mean(history_mass, late_response[None].expand_as(history_mass)),
        "history_mass_slope": _mean_slope(history_mass),
        "self_mass_mean": _finite_mean(self_mass),
        "self_mass_late": _masked_mean(self_mass, late_response[None].expand_as(self_mass)),
        "unresolved_mass_mean": _finite_mean(unresolved),
        "unresolved_mass_late": _masked_mean(unresolved, late_response[None].expand_as(unresolved)),
        "row_mass_conservation_error": _finite_mean((row_coverage - 1.0).abs()),
        "prompt_rows_with_mass_fraction": float((prompt_mass > 1e-12).float().mean().item()),
        "history_rows_with_mass_fraction": float((history_mass > 1e-12).float().mean().item()),
    }

    route = _role_pair_statistics(graph, field)
    feature.update(
        {
            "route_effective_sources_mean": _finite_mean(route["total_effective"]),
            "route_top1_share_mean": _finite_mean(route["total_top1"]),
            "route_mean_lag_fraction": _finite_mean(route["total_lag"]),
            "route_broad_shallow_mean": _finite_mean(
                route["total_effective"] * (1.0 - route["total_lag"])
            ),
            "prompt_effective_sources_mean": _finite_mean(route["prompt_effective"]),
            "history_effective_sources_mean": _finite_mean(route["history_effective"]),
            "history_effective_sources_late": _masked_mean(
                route["history_effective"],
                late_response[None].expand_as(route["history_effective"]),
            ),
            "history_top1_share_late": _masked_mean(
                route["history_top1"],
                late_response[None].expand_as(route["history_top1"]),
            ),
            "history_broad_shallow_late": _masked_mean(
                route["history_effective"] * (1.0 - route["history_lag"]),
                late_response[None].expand_as(route["history_effective"]),
            ),
            "prompt_code_effective_heads_mean": _finite_mean(
                route["prompt_head_effective"]
            ),
            "history_code_effective_heads_mean": _finite_mean(
                route["history_head_effective"]
            ),
            "prompt_code_valid_row_fraction": float(
                torch.isfinite(route["prompt_head_effective"]).float().mean().item()
            ),
            "history_code_valid_row_fraction": float(
                torch.isfinite(route["history_head_effective"]).float().mean().item()
            ),
            "prompt_observed_head_fraction": _finite_mean(route["prompt_observed"]),
            "history_observed_head_fraction": _finite_mean(route["history_observed"]),
        }
    )

    for mode in modes:
        state = _operator_role_state(
            graph,
            field,
            geometry,
            mode,
            seed=seed,
        )
        dispersion = state["dispersion"]
        valid = state["valid"]
        prompt_dispersion = dispersion[:, :, 0]
        history_dispersion = dispersion[:, :, 1]
        prompt_valid = valid[:, :, 0]
        history_valid = valid[:, :, 1]

        early_response_mask = early_response[None].expand_as(prompt_valid)
        late_response_mask = late_response[None].expand_as(prompt_valid)
        early_layer_mask = early_layer[:, None].expand_as(prompt_valid)
        late_layer_mask = late_layer[:, None].expand_as(prompt_valid)

        prompt_switch_valid = state["prompt_switch_valid"]
        history_switch_valid = state["history_switch_valid"]
        self_switch_valid = state["self_switch_valid"]
        if graph.response_count > 1:
            switch_position = torch.arange(1, graph.response_count)
            early_switch = switch_position < max(1, math.ceil(graph.response_count / 3))
            late_switch = switch_position >= max(
                graph.response_count - max(1, math.ceil(graph.response_count / 3)),
                1,
            )
            early_switch = early_switch[None].expand_as(prompt_switch_valid)
            late_switch = late_switch[None].expand_as(prompt_switch_valid)
        else:
            early_switch = torch.zeros_like(prompt_switch_valid)
            late_switch = torch.zeros_like(prompt_switch_valid)

        prefix = f"{mode}_"
        prompt_disp_early = _selected_mean(
            prompt_dispersion,
            prompt_valid & early_response_mask,
        )
        prompt_disp_late = _selected_mean(
            prompt_dispersion,
            prompt_valid & late_response_mask,
        )
        history_disp_early = _selected_mean(
            history_dispersion,
            history_valid & early_response_mask,
        )
        history_disp_late = _selected_mean(
            history_dispersion,
            history_valid & late_response_mask,
        )
        history_switch_late = _selected_mean(
            state["history_switch"],
            history_switch_valid & late_switch,
        )
        self_switch_late = _selected_mean(
            state["self_switch"],
            self_switch_valid & late_switch,
        )
        history_stability_late = 1.0 / (1.0 + history_switch_late)
        response_lockin = (
            feature["history_mass_late"]
            * feature["history_top1_share_late"]
            * history_stability_late
        )
        early_confusion = 0.5 * (prompt_disp_early + history_disp_early)
        late_confusion = 0.5 * (prompt_disp_late + history_disp_late)
        collapse = max(early_confusion - late_confusion, 0.0) * max(
            feature["history_mass_late"] - feature["history_mass_early"],
            0.0,
        )

        feature.update(
            {
                prefix + "prompt_dispersion_mean": _selected_mean(
                    prompt_dispersion, prompt_valid
                ),
                prefix + "prompt_dispersion_early": prompt_disp_early,
                prefix + "prompt_dispersion_late": prompt_disp_late,
                prefix + "history_dispersion_mean": _selected_mean(
                    history_dispersion, history_valid
                ),
                prefix + "history_dispersion_early": history_disp_early,
                prefix + "history_dispersion_late": history_disp_late,
                prefix + "history_dispersion_layer_shift": _selected_mean(
                    history_dispersion,
                    history_valid & late_layer_mask,
                )
                - _selected_mean(
                    history_dispersion,
                    history_valid & early_layer_mask,
                ),
                prefix + "prompt_history_distance_mean": _selected_mean(
                    state["prompt_history_distance"],
                    state["prompt_history_valid"],
                ),
                prefix + "prompt_history_distance_early": _selected_mean(
                    state["prompt_history_distance"],
                    state["prompt_history_valid"] & early_response_mask,
                ),
                prefix + "prompt_history_distance_late": _selected_mean(
                    state["prompt_history_distance"],
                    state["prompt_history_valid"] & late_response_mask,
                ),
                prefix + "prompt_step_switch_mean": _selected_mean(
                    state["prompt_switch"], prompt_switch_valid
                ),
                prefix + "prompt_step_switch_early": _selected_mean(
                    state["prompt_switch"], prompt_switch_valid & early_switch
                ),
                prefix + "history_step_switch_mean": _selected_mean(
                    state["history_switch"], history_switch_valid
                ),
                prefix + "history_step_switch_early": _selected_mean(
                    state["history_switch"], history_switch_valid & early_switch
                ),
                prefix + "history_step_switch_late": history_switch_late,
                prefix + "self_step_switch_late": self_switch_late,
                prefix + "history_stability_late": history_stability_late,
                prefix + "response_operator_lockin": response_lockin,
                prefix + "early_confusion_late_collapse": collapse,
            }
        )

    return feature
