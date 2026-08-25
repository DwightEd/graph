"""Flat all-layer baseline with the same attention values and no graph topology."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import HoloRouteConfig
from .detection import TokenResiduals
from .graph import EventGraph

FLAT_MODEL_TYPE = "flat1024"
FLAT_CHECKPOINT_SCHEMA = "holoroute-flat1024-checkpoint-v2"
FLAT_RESIDUAL_NAMES = ("flat1024",)


@dataclass(frozen=True)
class Pairs:
    graph: EventGraph
    source: torch.Tensor
    target: torch.Tensor
    role: torch.Tensor
    lag: torch.Tensor
    value: torch.Tensor
    observed: torch.Tensor
    present: torch.Tensor
    event_pair: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.source.numel())

    @property
    def device(self) -> torch.device:
        return self.value.device

    @property
    def query(self) -> torch.Tensor:
        return self.target - self.graph.response_start


@dataclass(frozen=True)
class PairPrediction:
    value: torch.Tensor
    support: torch.Tensor


@torch.no_grad()
def build_pairs(graph: EventGraph) -> Pairs:
    if not graph.event_count:
        empty = torch.empty(0, dtype=torch.long, device=graph.device)
        return Pairs(
            graph=graph,
            source=empty,
            target=empty,
            role=empty,
            lag=empty,
            value=graph.events.value.new_empty((0, graph.layer_count, graph.head_count)),
            observed=torch.empty((0, graph.layer_count, graph.head_count), dtype=torch.bool, device=graph.device),
            present=torch.empty((0, graph.layer_count), dtype=torch.bool, device=graph.device),
            event_pair=empty,
        )

    key = graph.events.source * graph.token_count + graph.events.target
    unique, event_pair = torch.unique(key, sorted=True, return_inverse=True)
    source = torch.div(unique, graph.token_count, rounding_mode="floor")
    target = unique.remainder(graph.token_count)
    pair_count = len(unique)

    value = graph.events.value.new_zeros((pair_count, graph.layer_count, graph.head_count))
    observed = torch.zeros_like(value, dtype=torch.bool)
    present = torch.zeros((pair_count, graph.layer_count), dtype=torch.bool, device=graph.device)
    value[event_pair, graph.events.layer] = graph.events.value
    observed[event_pair, graph.events.layer] = graph.events.observed
    present[event_pair, graph.events.layer] = True

    return Pairs(
        graph=graph,
        source=source,
        target=target,
        role=(source >= graph.response_start).long(),
        lag=target - source,
        value=value,
        observed=observed,
        present=present,
        event_pair=event_pair,
    )


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.network(state)


class Flat1024(nn.Module):
    """Residual MLP over one flattened ``layer x head`` tensor per token pair."""

    def __init__(
        self,
        layers: int,
        heads: int,
        hidden: int = 96,
        blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = int(layers)
        self.heads = int(heads)
        flat = self.layers * self.heads
        self.value_projection = nn.Linear(flat, hidden)
        self.observation_projection = nn.Linear(flat, hidden)
        self.presence_projection = nn.Linear(self.layers, hidden)
        self.metadata = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(ResidualBlock(hidden, dropout) for _ in range(blocks))
        self.value_decoder = nn.Linear(hidden, flat)
        self.support_decoder = nn.Linear(hidden, flat)

    def forward(
        self,
        pairs: Pairs,
        values: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
    ) -> PairPrediction:
        values = pairs.value if values is None else values
        observed = pairs.observed if observed is None else observed
        query_position = pairs.query.float() / max(pairs.graph.response_count - 1, 1)
        source_position = pairs.source.float() / max(pairs.graph.token_count - 1, 1)
        lag = torch.log1p(pairs.lag.float()) / np.log1p(max(pairs.graph.token_count, 2))
        metadata = torch.stack((query_position, source_position, lag, pairs.role.float()), dim=-1)

        state = (
            self.value_projection(torch.log1p(values).flatten(1))
            + self.observation_projection(observed.float().flatten(1))
            + self.presence_projection(pairs.present.float())
            + self.metadata(metadata)
        )
        state = self.norm(state)
        for block in self.blocks:
            state = block(state)
        shape = (pairs.count, self.layers, self.heads)
        return PairPrediction(
            value=F.softplus(self.value_decoder(state)).reshape(shape),
            support=self.support_decoder(state).reshape(shape),
        )


def choose_layer_blocks(
    pairs: Pairs,
    fraction: float,
    minimum: int,
    generator: torch.Generator,
) -> torch.Tensor:
    eligible = torch.nonzero(pairs.present, as_tuple=False)
    selected = torch.zeros_like(pairs.present)
    if not len(eligible):
        return selected
    count = min(len(eligible), max(int(round(len(eligible) * fraction)), minimum))
    order = torch.randperm(len(eligible), generator=generator, device=pairs.device)
    blocks = eligible[order[:count]]
    selected[blocks[:, 0], blocks[:, 1]] = True
    return selected


def mask_layer_blocks(pairs: Pairs, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = pairs.value.clone()
    observed = pairs.observed.clone()
    values[selected] = 0.0
    observed[selected] = False
    return values, observed


def block_error(
    prediction: PairPrediction,
    pairs: Pairs,
    config: HoloRouteConfig,
) -> torch.Tensor:
    observed = pairs.observed.float()
    count = observed.sum(dim=-1).clamp_min(1.0)
    value = F.smooth_l1_loss(
        torch.log1p(prediction.value),
        torch.log1p(pairs.value),
        reduction="none",
    )
    value = (value * observed).sum(dim=-1) / count

    support = F.binary_cross_entropy_with_logits(
        prediction.support,
        observed,
        reduction="none",
    ).mean(dim=-1)

    censored = 1.0 - observed
    bound = F.relu(prediction.value - pairs.graph.attention_floor).square()
    bound = (bound * censored).sum(dim=-1) / censored.sum(dim=-1).clamp_min(1.0)
    return value + config.loss.support * support + config.loss.censored * bound


def flat_loss(
    model: Flat1024,
    pairs: Pairs,
    config: HoloRouteConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = choose_layer_blocks(
        pairs,
        config.train.mask_fraction,
        config.train.minimum_masked_events,
        generator,
    )
    values, observed = mask_layer_blocks(pairs, selected)
    prediction = model(pairs, values, observed)
    error = block_error(prediction, pairs, config)
    return error[selected].mean() if bool(selected.any()) else prediction.value.sum() * 0.0


def layer_schedule(pairs: Pairs, folds: int, seed: int) -> list[torch.Tensor]:
    eligible = torch.nonzero(pairs.present, as_tuple=False)
    if not len(eligible):
        return []
    generator = torch.Generator(device=pairs.device).manual_seed(int(seed))
    order = eligible[torch.randperm(len(eligible), generator=generator, device=pairs.device)]
    fold_count = max(1, min(int(folds), len(eligible)))
    schedule = []
    for blocks in torch.tensor_split(order, fold_count):
        if not len(blocks):
            continue
        selected = torch.zeros_like(pairs.present)
        selected[blocks[:, 0], blocks[:, 1]] = True
        schedule.append(selected)
    return schedule


@torch.no_grad()
def score_flat(
    model: Flat1024,
    pairs: Pairs,
    config: HoloRouteConfig,
    seed: int,
) -> TokenResiduals:
    model.eval()
    pair_error = pairs.value.new_full((pairs.count, pairs.graph.layer_count), torch.nan)
    for selected in layer_schedule(pairs, config.detection.score_folds, seed):
        values, observed = mask_layer_blocks(pairs, selected)
        prediction = model(pairs, values, observed)
        pair_error[selected] = block_error(prediction, pairs, config)[selected]

    total = pairs.value.new_zeros(pairs.graph.response_count)
    count = pairs.value.new_zeros(pairs.graph.response_count)
    available = torch.isfinite(pair_error) & pairs.present
    if bool(available.any()):
        pair, _ = torch.nonzero(available, as_tuple=True)
        value = pair_error[available]
        token = pairs.query[pair]
        total.index_add_(0, token, value)
        count.index_add_(0, token, torch.ones_like(value))

    residual = pairs.value.new_full((pairs.graph.response_count, 1), torch.nan)
    valid = count > 0
    residual[valid, 0] = total[valid] / count[valid]
    return TokenResiduals(
        value=residual.cpu().numpy().astype(np.float32),
        coverage=count[:, None].cpu().numpy().astype(np.float32),
    )
