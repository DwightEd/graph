"""Source-resolved path flow on exact functional token messages.

The input is the all-source generation DAG saved by the functional-message
observer.  Edges are never learned and hallucination labels are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

SOURCE_GROUPS = ("evidence", "other_prompt", "response")
EVIDENCE, OTHER_PROMPT, RESPONSE = range(3)
FLOW_CHANNELS = (
    "positive_function",
    "negative_function",
    "attention",
    "residual_message_norm",
)


@dataclass(frozen=True)
class GroundedFlow:
    """Per-target path origins and response-anchor mediation."""

    response_seeded_path_share: torch.Tensor  # [R]
    response_seeded_anchor_flow: torch.Tensor  # [R]
    source_path_posterior: torch.Tensor  # [R, evidence/question/response]
    direct_response_share: torch.Tensor  # [R]
    gather_distance: torch.Tensor  # [R]
    anchor_occupancy: torch.Tensor  # [target, response anchor]
    anchor_group_occupancy: torch.Tensor  # [target, source group, response anchor]
    future_anchor_influence: torch.Tensor  # [response anchor]
    anchor_concentration: torch.Tensor  # [target]
    dominant_anchor: torch.Tensor  # [target], response index or -1
    valid: torch.Tensor  # [R]
    anchor_valid: torch.Tensor  # [R]


def token_transition(
    token_flow: torch.Tensor,
    response_start: int,
    channel: str = "positive_function",
) -> torch.Tensor:
    """Normalize one capacity channel into a source-to-response transition.

    The returned matrix has shape ``[all source tokens, response targets]``.
    Column ``r`` contains the normalized incoming capacity of token
    ``response_start + r``.  Non-causal cells and undefined columns are zero.
    """

    if channel not in FLOW_CHANNELS:
        raise ValueError(f"unknown flow channel: {channel}")
    flow = torch.as_tensor(token_flow, dtype=torch.float64)
    if flow.ndim != 3 or flow.shape[-1] != len(FLOW_CHANNELS):
        raise ValueError("token_flow must be [response, token, flow_channel]")
    response, tokens, _ = flow.shape
    if not 0 < response_start <= tokens or response != tokens - response_start:
        raise ValueError("token_flow does not align with response_start")

    target = torch.arange(response_start, tokens)[:, None]
    source = torch.arange(tokens)[None]
    capacity = flow[..., FLOW_CHANNELS.index(channel)].clamp_min(0)
    capacity = capacity * (source < target)
    total = capacity.sum(dim=1, keepdim=True)
    normalized = torch.where(total > 0, capacity / total.clamp_min(1e-300), 0)
    return normalized.T.contiguous()


def path_closure(response_transition: torch.Tensor) -> torch.Tensor:
    """Sum every directed path product in a causal response DAG."""

    if response_transition.ndim != 2 or response_transition.shape[0] != response_transition.shape[1]:
        raise ValueError("response transition must be square")
    if torch.count_nonzero(torch.tril(response_transition)):
        raise ValueError("response transition must be strictly causal")
    identity = torch.eye(
        len(response_transition),
        dtype=response_transition.dtype,
        device=response_transition.device,
    )
    return torch.linalg.solve_triangular(
        identity - response_transition,
        identity,
        upper=True,
    )


def prompt_paths(
    transition: torch.Tensor,
    response_closure: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    """Average prompt-source path mass, retaining the common prompt prior."""

    prompt_count = len(source_mask)
    if not source_mask.any():
        return torch.zeros(
            response_closure.shape[1],
            dtype=response_closure.dtype,
            device=response_closure.device,
        )
    direct = transition[:prompt_count][source_mask].sum(dim=0) / prompt_count
    return direct @ response_closure


def analyze_flow(
    graph: Mapping[str, Any],
    *,
    channel: str = "positive_function",
    gather_window: int = 64,
    future_window: int = 64,
) -> GroundedFlow:
    """Trace prompt- and response-seeded paths through response anchors.

    For targets with prior response tokens, a fixed binary prior assigns equal
    total mass to prompt tokens and earlier response tokens, uniformly within
    each side.  Conditioning this prior on reaching the target removes the
    ordinary group-size advantage.  Response zero-hop starts are subtracted
    before anchor mediation is measured, so a direct dependency is not
    mislabeled as a multi-hop anchor route.
    """

    token_flow = torch.as_tensor(graph["token_flow"])
    response_start = int(graph["response_start"])
    evidence_mask = torch.as_tensor(graph["evidence_mask"], dtype=torch.bool)
    response, tokens, _ = token_flow.shape
    if evidence_mask.shape != (response_start,):
        raise ValueError("evidence_mask must align with the prompt")
    if gather_window <= 0 or future_window <= 0:
        raise ValueError("flow windows must be positive")

    transition = token_transition(token_flow, response_start, channel)
    response_closure = path_closure(transition[response_start:])
    evidence_paths = prompt_paths(transition, response_closure, evidence_mask)
    question_paths = prompt_paths(transition, response_closure, ~evidence_mask)
    response_prefix_paths = response_closure.cumsum(dim=0)

    path_share = torch.full((response,), torch.nan, dtype=torch.float32)
    anchor_flow = torch.full((response,), torch.nan, dtype=torch.float32)
    source_posterior = torch.zeros(response, len(SOURCE_GROUPS), dtype=torch.float32)
    direct_response = torch.full((response,), torch.nan, dtype=torch.float32)
    gather_distance = torch.full((response,), torch.nan, dtype=torch.float32)
    anchor_occupancy = torch.zeros(response, response, dtype=torch.float32)
    anchor_group = torch.zeros(response, len(SOURCE_GROUPS), response, dtype=torch.float32)
    future_anchor = torch.full((response,), torch.nan, dtype=torch.float32)
    anchor_concentration = torch.full((response,), torch.nan, dtype=torch.float32)
    dominant_anchor = torch.full((response,), -1, dtype=torch.int64)
    valid = torch.zeros(response, dtype=torch.bool)
    anchor_valid = torch.zeros(response, dtype=torch.bool)

    for offset in range(response):
        target = response_start + offset
        incoming = transition[:target, offset]
        if incoming.sum() > 0:
            lag = target - torch.arange(target, dtype=torch.float64)
            gather_distance[offset] = (
                incoming * lag.clamp_max(gather_window)
            ).sum().div(gather_window).float()
            direct_response[offset] = incoming[response_start:target].sum().float()

        if offset:
            prompt_weight = response_weight = 0.5
            response_paths = response_prefix_paths[offset - 1] / offset
        else:
            prompt_weight, response_weight = 1.0, 0.0
            response_paths = torch.zeros(response, dtype=torch.float64)

        forward = torch.stack(
            (
                prompt_weight * evidence_paths,
                prompt_weight * question_paths,
                response_weight * response_paths,
            )
        )
        partition = forward[:, offset]
        total_partition = partition.sum()
        if total_partition <= 0:
            continue

        posterior = partition / total_partition
        source_posterior[offset] = posterior.float()
        path_share[offset] = posterior[RESPONSE].float()
        valid[offset] = True

        # Every target-reaching path that visits response node v factorizes as
        # source->v times v->target.  Subtract paths whose sampled source is v
        # itself; what remains is genuine transit through the anchor.
        occupancy = (
            forward[:, : offset + 1]
            * response_closure[: offset + 1, offset]
            / total_partition
        )
        source_start = torch.zeros_like(occupancy)
        if response_weight:
            source_start[RESPONSE, :offset] = (
                (response_weight / offset)
                * response_closure[:offset, offset]
                / total_partition
            )
        transit = (occupancy - source_start).clamp_min(0)[:, :offset]
        transit_mass = transit.sum()
        if transit_mass <= 0:
            continue

        group_mass = transit.sum(dim=1)
        anchor_flow[offset] = (group_mass[RESPONSE] / transit_mass).float()
        per_anchor = transit.sum(dim=0)
        normalized = per_anchor / transit_mass
        anchor_occupancy[offset, :offset] = normalized.float()
        anchor_group[offset, :, :offset] = (transit / transit_mass).float()
        anchor_concentration[offset] = normalized.max().float()
        dominant_anchor[offset] = int(normalized.argmax())
        anchor_valid[offset] = True

    for anchor in range(response):
        stop = min(response, anchor + future_window + 1)
        influence = anchor_occupancy[anchor + 1 : stop, anchor]
        if len(influence):
            future_anchor[anchor] = influence.mean()

    return GroundedFlow(
        response_seeded_path_share=path_share,
        response_seeded_anchor_flow=anchor_flow,
        source_path_posterior=source_posterior,
        direct_response_share=direct_response,
        gather_distance=gather_distance,
        anchor_occupancy=anchor_occupancy,
        anchor_group_occupancy=anchor_group,
        future_anchor_influence=future_anchor,
        anchor_concentration=anchor_concentration,
        dominant_anchor=dominant_anchor,
        valid=valid,
        anchor_valid=anchor_valid,
    )


def flow_to_dict(flow: GroundedFlow) -> dict[str, torch.Tensor]:
    return {name: getattr(flow, name) for name in GroundedFlow.__dataclass_fields__}
