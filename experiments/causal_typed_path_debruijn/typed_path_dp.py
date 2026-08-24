"""Channel-resolved typed token-path dynamic programming.

This is the token-path ablation and explanation view.  It walks attention
provenance backward along exact retained RR endpoints while preserving every
layer/head channel.  It is not a reconstruction of physical transformer
computation because ``W_V``, ``W_O``, residual, and MLP paths are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PathConfig
from .graph_builder import CausalRoutingGraph, RP, RR_FAR, RR_NEAR


PROMPT_EXIT = 0
SELF_EXIT = 1
UNRESOLVED_EXIT = 2
SINK_NAMES = ("prompt", "self", "unresolved")

DIRECT_PROMPT = 0
RELAYED_PROMPT = 1
SELF_ROUTE = 2
UNRESOLVED_ROUTE = 3
NEAR_CLOSED = 4
FAR_CLOSED = 5
MIXED_CLOSED = 6
ROUTE_NAMES = (
    "direct_prompt",
    "relayed_prompt",
    "self_exit",
    "unresolved_exit",
    "near_closed",
    "far_closed",
    "mixed_closed",
)


@dataclass(frozen=True)
class TypedPathResult:
    """Mass partition induced by at most ``max_hops`` RR transitions.

    ``exit_mass[t,c,h,s]`` is the mass that first reaches sink ``s`` after
    ``h+1`` row evaluations. ``survival_pattern`` holds paths that traverse all
    ``K`` RR edges. Pattern bits are encoded *forward*, older-to-current, with
    near=0 and far=1.
    """

    exit_mass: torch.Tensor
    survival_pattern: torch.Tensor
    route_distribution: torch.Tensor
    prompt_lineage_mass: torch.Tensor
    response_survival: torch.Tensor
    conservation_error: torch.Tensor
    max_hops: int
    route_names: tuple[str, ...] = ROUTE_NAMES

    @property
    def num_response_tokens(self) -> int:
        return int(self.route_distribution.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.route_distribution.shape[1])

    def validate(self) -> "TypedPathResult":
        r, c, states = self.route_distribution.shape
        if states != len(ROUTE_NAMES):
            raise ValueError("route_distribution must use the seven fixed states")
        if self.exit_mass.shape != (r, c, self.max_hops, len(SINK_NAMES)):
            raise ValueError("exit_mass has the wrong shape")
        if self.survival_pattern.shape != (r, c, 2**self.max_hops):
            raise ValueError("survival_pattern has the wrong shape")
        for tensor in (
            self.route_distribution,
            self.exit_mass,
            self.survival_pattern,
            self.prompt_lineage_mass,
            self.response_survival,
            self.conservation_error,
        ):
            if tensor.device != self.route_distribution.device:
                raise ValueError("typed-path tensors must share one device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("typed-path tensors must be finite")
        if (
            self.prompt_lineage_mass.shape != (r, c)
            or self.response_survival.shape != (r, c)
        ):
            raise ValueError("typed-path summaries must be [R,C]")
        if self.conservation_error.shape != (r, c):
            raise ValueError("conservation_error must be [R,C]")
        if bool((self.route_distribution < -2e-6).any()):
            raise ValueError("route masses must be non-negative")
        if float(self.conservation_error.max().item()) > 1e-4:
            raise ValueError("typed-path recurrence failed mass conservation")
        return self


def _scatter_edge_messages(
    values: torch.Tensor,
    *,
    target: torch.Tensor,
    channel: torch.Tensor,
    response_count: int,
    num_channels: int,
) -> torch.Tensor:
    """Sum ``[E,D]`` messages into a dense ``[R,C,D]`` state."""

    output = torch.zeros(
        (response_count * num_channels, values.shape[-1]),
        dtype=values.dtype,
        device=values.device,
    )
    flat_target = target * num_channels + channel
    output.index_add_(0, flat_target, values)
    return output.reshape(response_count, num_channels, values.shape[-1])


@torch.no_grad()
def typed_path_dp(
    graph: CausalRoutingGraph,
    *,
    config: PathConfig | None = None,
) -> TypedPathResult:
    """Compute a sparse, parameter-free, exactly mass-conserving path view."""

    graph.validate()
    config = PathConfig() if config is None else config
    config.validate()
    hops = int(config.max_hops)
    response_count = graph.num_response_tokens
    channels = graph.num_channels
    dtype = graph.weight.dtype
    device = graph.device

    local_exit = torch.stack(
        (graph.prompt_channel, graph.self_channel, graph.unresolved_channel),
        dim=-1,
    )
    exit_mass = torch.zeros(
        (response_count, channels, hops, len(SINK_NAMES)),
        dtype=dtype,
        device=device,
    )
    exit_mass[:, :, 0] = local_exit

    rr = graph.relation != RP
    rr_source = graph.source[rr] - graph.response_idx
    rr_target = graph.query[rr]
    rr_channel = graph.edge_channel[rr]
    rr_weight = graph.weight[rr]
    rr_bit = (graph.relation[rr] == RR_FAR).long()

    # Exit at hop h is an RR transition into an earlier token followed by the
    # exit distribution already computed there at hop h-1.
    for hop in range(1, hops):
        if rr_weight.numel():
            message = (
                exit_mass[rr_source, rr_channel, hop - 1]
                * rr_weight.unsqueeze(-1)
            )
            exit_mass[:, :, hop] = _scatter_edge_messages(
                message,
                target=rr_target,
                channel=rr_channel,
                response_count=response_count,
                num_channels=channels,
            )

    # A path observed backward as current->older is appended on the right of
    # the forward older->current code. Thus prior_pattern*2 + current_edge_bit.
    patterns = torch.empty(0, dtype=dtype, device=device)
    for hop in range(hops):
        width = 2 ** (hop + 1)
        current = torch.zeros(
            (response_count * channels, width), dtype=dtype, device=device
        )
        if rr_weight.numel():
            if hop == 0:
                message = rr_weight.unsqueeze(-1)
                pattern_index = rr_bit.unsqueeze(-1)
            else:
                previous = patterns[rr_source, rr_channel]
                message = previous * rr_weight.unsqueeze(-1)
                base = torch.arange(
                    previous.shape[-1], dtype=torch.long, device=device
                ).unsqueeze(0)
                pattern_index = base * 2 + rr_bit.unsqueeze(-1)
            flat_row = rr_target * channels + rr_channel
            expanded_row = flat_row.unsqueeze(-1).expand_as(pattern_index)
            current.index_put_(
                (expanded_row.reshape(-1), pattern_index.reshape(-1)),
                message.reshape(-1),
                accumulate=True,
            )
        patterns = current.reshape(response_count, channels, width)

    survival_pattern = patterns
    response_survival = survival_pattern.sum(dim=-1)
    prompt_lineage_mass = exit_mass[..., PROMPT_EXIT].sum(dim=2)
    self_exit = exit_mass[..., SELF_EXIT].sum(dim=2)
    unresolved_exit = exit_mass[..., UNRESOLVED_EXIT].sum(dim=2)
    direct_prompt = exit_mass[:, :, 0, PROMPT_EXIT]
    relayed_prompt = prompt_lineage_mass - direct_prompt
    near_closed = survival_pattern[..., 0]
    far_closed = survival_pattern[..., -1]
    mixed_closed = (response_survival - near_closed - far_closed).clamp_min(0.0)
    route_distribution = torch.stack(
        (
            direct_prompt,
            relayed_prompt,
            self_exit,
            unresolved_exit,
            near_closed,
            far_closed,
            mixed_closed,
        ),
        dim=-1,
    ).clamp_min(0.0)
    conservation_error = (route_distribution.sum(dim=-1) - 1.0).abs()
    return TypedPathResult(
        exit_mass=exit_mass,
        survival_pattern=survival_pattern,
        route_distribution=route_distribution,
        prompt_lineage_mass=prompt_lineage_mass,
        response_survival=response_survival,
        conservation_error=conservation_error,
        max_hops=hops,
    ).validate()
