"""Role-aware neighbourhood aggregation for token attention graphs."""

from dataclasses import dataclass

import torch
from torch import nn

PROMPT_ROLE = 0
RESPONSE_ROLE = 1
ROLE_COUNT = 2


@dataclass(frozen=True)
class RouteMoments:
    mean: torch.Tensor
    spread: torch.Tensor
    mass: torch.Tensor


def lag_bucket(lag: torch.Tensor, bucket_count: int) -> torch.Tensor:
    return torch.floor(torch.log2(lag.float().clamp_min(1))).long().clamp_max(
        bucket_count - 1
    )


def route_moments(
    message: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    head: torch.Tensor,
    role: torch.Tensor,
    response_count: int,
    head_count: int,
) -> RouteMoments:
    """Return weighted mean, spread and mass for prompt/response neighbours."""

    hidden = message.shape[-1]
    group_count = response_count * head_count * ROLE_COUNT
    group = (target * head_count + head) * ROLE_COUNT + role

    mass = weight.new_zeros(group_count)
    first = message.new_zeros((group_count, hidden))
    second = message.new_zeros((group_count, hidden))

    mass.index_add_(0, group, weight)
    first.index_add_(0, group, message * weight[:, None])
    second.index_add_(0, group, message.square() * weight[:, None])

    denominator = mass.clamp_min(1e-8)[:, None]
    mean = first / denominator
    variance = (second / denominator - mean.square()).clamp_min(0.0)
    spread = torch.sqrt(variance + 1e-8)
    spread = torch.where(mass[:, None] > 0, spread, torch.zeros_like(spread))

    shape = (response_count, head_count, ROLE_COUNT)
    return RouteMoments(
        mean=mean.reshape(*shape, hidden),
        spread=spread.reshape(*shape, hidden),
        mass=mass.reshape(shape),
    )


class RouteAggregator(nn.Module):
    """Fuse role-separated neighbour moments into one update per token."""

    def __init__(self, layers: int, heads: int, hidden: int) -> None:
        super().__init__()
        summary_dim = hidden * 4 + 2
        self.route_fusion = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.self_message = nn.Linear(hidden, hidden)
        self.unresolved_message = nn.Parameter(torch.empty(layers, heads, hidden))
        self.head_query = nn.Linear(hidden, hidden, bias=False)
        self.head_key = nn.Linear(hidden, hidden, bias=False)
        self.head_score = nn.Linear(hidden, 1, bias=False)
        nn.init.normal_(self.unresolved_message, std=0.02)

    def forward(
        self,
        response_state: torch.Tensor,
        message: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        head: torch.Tensor,
        role: torch.Tensor,
        diagonal: torch.Tensor,
        unresolved: torch.Tensor,
        head_identity: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        moments = route_moments(
            message,
            weight,
            target,
            head,
            role,
            len(response_state),
            diagonal.shape[1],
        )

        prompt_mean = moments.mean[:, :, PROMPT_ROLE]
        prompt_spread = moments.spread[:, :, PROMPT_ROLE]
        response_mean = moments.mean[:, :, RESPONSE_ROLE]
        response_spread = moments.spread[:, :, RESPONSE_ROLE]
        prompt_mass = torch.log1p(moments.mass[:, :, PROMPT_ROLE])
        response_mass = torch.log1p(moments.mass[:, :, RESPONSE_ROLE])

        summary = torch.cat(
            (
                prompt_mean,
                prompt_spread,
                response_mean,
                response_spread,
                prompt_mass[..., None],
                response_mass[..., None],
            ),
            dim=-1,
        )
        cells = self.route_fusion(summary)
        cells = cells + diagonal[..., None] * self.self_message(response_state)[:, None]
        cells = cells + unresolved[..., None] * self.unresolved_message[layer][None]
        cells = cells + head_identity[None]

        query = self.head_query(response_state)[:, None]
        score = self.head_score(torch.tanh(query + self.head_key(cells))).squeeze(-1)
        attention = torch.softmax(score, dim=-1)
        return (cells * attention[..., None]).sum(dim=1)
