"""Flat all-layer attention baseline for HoloRoute.

The baseline keeps the same layer-by-head values as HoloRoute but removes every
adjacency relation. One sample for the MLP is an exact ``(source, target)``
token pair with a dense ``[layer, head]`` value/mask tensor. Training masks
complete layer blocks and reconstructs them from the remaining coordinates of
the same pair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments.attention_holonomy_audit.graph import AttentionEventGraph
from .config import DensityConfig, TrainConfig

FLAT_SCORE_FEATURES = ("flat_1024_reconstruction",)
FLAT_CHECKPOINT_SCHEMA = "holoroute-flat1024-checkpoint-v1"
FLAT_MODEL_TYPE = "flat_1024"


@dataclass(frozen=True)
class Flat1024ModelConfig:
    hidden_dim: int = 96
    blocks: int = 3
    dropout: float = 0.1


@dataclass(frozen=True)
class Flat1024MaskConfig:
    train_fraction: float = 0.2
    minimum_blocks: int = 1
    score_rounds: int = 8


@dataclass(frozen=True)
class Flat1024Config:
    model: Flat1024ModelConfig = field(default_factory=Flat1024ModelConfig)
    masking: Flat1024MaskConfig = field(default_factory=Flat1024MaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    density: DensityConfig = field(default_factory=DensityConfig)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FlatPairView:
    sample_id: str
    source_id: str
    task_type: str
    response_idx: int
    num_tokens: int
    num_response_tokens: int
    num_layers: int
    num_heads: int
    attention_floor: float
    pair_source: torch.Tensor
    pair_target: torch.Tensor
    pair_role: torch.Tensor
    pair_lag: torch.Tensor
    value: torch.Tensor
    observed: torch.Tensor
    layer_present: torch.Tensor
    event_pair: torch.Tensor
    event_layer: torch.Tensor
    response_token_ids: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.value.device

    @property
    def num_pairs(self) -> int:
        return int(self.pair_source.numel())

    @property
    def pair_query(self) -> torch.Tensor:
        return self.pair_target - self.response_idx

    @property
    def flat_dim(self) -> int:
        return self.num_layers * self.num_heads

    def validate(self) -> "FlatPairView":
        pairs = self.num_pairs
        if self.value.shape != (pairs, self.num_layers, self.num_heads):
            raise ValueError("flat pair values must be [pair, layer, head]")
        if self.observed.shape != self.value.shape:
            raise ValueError("flat pair observation mask must match values")
        if self.layer_present.shape != (pairs, self.num_layers):
            raise ValueError("layer_present must be [pair, layer]")
        if self.event_pair.shape != self.event_layer.shape:
            raise ValueError("event-to-pair map must align with event layers")
        if pairs and bool((self.pair_source >= self.pair_target).any()):
            raise ValueError("flat pair view must remain prefix-causal")
        return self


@torch.no_grad()
def build_flat_pair_view(graph: AttentionEventGraph) -> FlatPairView:
    """Group layer-specific events into exact pair-level ``[L,H]`` tensors."""

    if graph.num_events == 0:
        device = graph.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        return FlatPairView(
            sample_id=graph.sample_id,
            source_id=graph.source_id,
            task_type=graph.task_type,
            response_idx=graph.response_idx,
            num_tokens=graph.num_tokens,
            num_response_tokens=graph.num_response_tokens,
            num_layers=graph.num_layers,
            num_heads=graph.num_heads,
            attention_floor=graph.attention_floor,
            pair_source=empty,
            pair_target=empty,
            pair_role=empty,
            pair_lag=empty,
            value=torch.empty((0, graph.num_layers, graph.num_heads), device=device),
            observed=torch.empty((0, graph.num_layers, graph.num_heads), dtype=torch.bool, device=device),
            layer_present=torch.empty((0, graph.num_layers), dtype=torch.bool, device=device),
            event_pair=empty,
            event_layer=empty,
            response_token_ids=graph.response_token_ids,
        ).validate()

    pair_key = graph.event_source * graph.num_tokens + graph.event_target
    unique_key, event_pair = torch.unique(pair_key, sorted=True, return_inverse=True)
    pair_source = torch.div(unique_key, graph.num_tokens, rounding_mode="floor")
    pair_target = unique_key.remainder(graph.num_tokens)
    pairs = len(unique_key)
    value = graph.event_head_value.new_zeros((pairs, graph.num_layers, graph.num_heads))
    observed = torch.zeros_like(value, dtype=torch.bool)
    layer_present = torch.zeros((pairs, graph.num_layers), dtype=torch.bool, device=graph.device)
    value[event_pair, graph.event_layer] = graph.event_head_value
    observed[event_pair, graph.event_layer] = graph.event_head_observed
    layer_present[event_pair, graph.event_layer] = True
    return FlatPairView(
        sample_id=graph.sample_id,
        source_id=graph.source_id,
        task_type=graph.task_type,
        response_idx=graph.response_idx,
        num_tokens=graph.num_tokens,
        num_response_tokens=graph.num_response_tokens,
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        attention_floor=graph.attention_floor,
        pair_source=pair_source,
        pair_target=pair_target,
        pair_role=(pair_source >= graph.response_idx).long(),
        pair_lag=pair_target - pair_source,
        value=value,
        observed=observed,
        layer_present=layer_present,
        event_pair=event_pair,
        event_layer=graph.event_layer,
        response_token_ids=graph.response_token_ids,
    ).validate()


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.body(state)


class Flat1024Model(nn.Module):
    """MLP over a flattened ``L x H`` pair tensor without graph adjacency."""

    def __init__(self, num_layers: int, num_heads: int, config: Flat1024ModelConfig | None = None) -> None:
        super().__init__()
        self.config = Flat1024ModelConfig() if config is None else config
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        flat = self.num_layers * self.num_heads
        hidden = self.config.hidden_dim
        self.value_projection = nn.Linear(flat, hidden)
        self.observation_projection = nn.Linear(flat, hidden)
        self.layer_presence_projection = nn.Linear(self.num_layers, hidden)
        self.metadata = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden, self.config.dropout) for _ in range(self.config.blocks))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, flat),
        )

    def forward(self, view: FlatPairView, *, value: torch.Tensor | None = None, observed: torch.Tensor | None = None) -> torch.Tensor:
        current_value = view.value if value is None else value
        current_observed = view.observed if observed is None else observed
        values = torch.log1p(current_value).flatten(1)
        mask = current_observed.float().flatten(1)
        query_position = view.pair_query.float() / max(view.num_response_tokens - 1, 1)
        source_position = view.pair_source.float() / max(view.num_tokens - 1, 1)
        lag = torch.log1p(view.pair_lag.float()) / np.log1p(max(view.num_tokens, 2))
        metadata = torch.stack((query_position, source_position, lag, view.pair_role.float()), dim=-1)
        state = (
            self.value_projection(values)
            + self.observation_projection(mask)
            + self.layer_presence_projection(view.layer_present.float())
            + self.metadata(metadata)
        )
        state = self.input_norm(state)
        for block in self.blocks:
            state = block(state)
        prediction = F.softplus(self.decoder(state))
        return prediction.reshape(view.num_pairs, view.num_layers, view.num_heads)


def sample_layer_block_mask(view: FlatPairView, *, fraction: float, minimum: int, generator: torch.Generator) -> torch.Tensor:
    """Mask existing pair-layer blocks while leaving one layer visible per pair."""

    eligible = torch.nonzero(view.layer_present, as_tuple=False)
    mask = torch.zeros_like(view.layer_present)
    if len(eligible) == 0:
        return mask
    count = max(int(round(len(eligible) * float(fraction))), int(minimum))
    count = min(count, len(eligible))
    order = torch.randperm(len(eligible), generator=generator, device=view.device)
    selected = eligible[order[:count]]
    mask[selected[:, 0], selected[:, 1]] = True
    fully_masked = mask.sum(dim=1) >= view.layer_present.sum(dim=1)
    for pair in torch.nonzero(fully_masked, as_tuple=False).flatten().tolist():
        layers = torch.nonzero(mask[pair], as_tuple=False).flatten()
        keep = layers[torch.randint(len(layers), (1,), generator=generator, device=view.device)]
        mask[pair, keep] = False
    if not bool(mask.any()):
        first = eligible[order[0]]
        mask[first[0], first[1]] = True
    return mask


def apply_layer_block_mask(view: FlatPairView, block_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = view.value.clone()
    observed = view.observed.clone()
    value[block_mask] = 0.0
    observed[block_mask] = False
    return value, observed


def block_vector_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    huber = F.smooth_l1_loss(torch.log1p(prediction), torch.log1p(target), reduction="none").mean(dim=-1)
    return cosine.clamp_min(0.0) + huber


def flat1024_loss(model: Flat1024Model, view: FlatPairView, config: Flat1024Config, *, generator: torch.Generator) -> torch.Tensor:
    block_mask = sample_layer_block_mask(
        view,
        fraction=config.masking.train_fraction,
        minimum=config.masking.minimum_blocks,
        generator=generator,
    )
    value, observed = apply_layer_block_mask(view, block_mask)
    prediction = model(view, value=value, observed=observed)
    error = block_vector_error(prediction[block_mask], view.value[block_mask])
    return error.mean() if len(error) else prediction.sum() * 0.0


def _scoring_schedule(view: FlatPairView, rounds: int, seed: int) -> list[torch.Tensor]:
    eligible = torch.nonzero(view.layer_present, as_tuple=False)
    if len(eligible) == 0:
        return []
    folds = max(1, min(int(rounds), int(view.num_layers)))
    schedule = [torch.zeros_like(view.layer_present) for _ in range(folds)]
    generator = torch.Generator(device=view.device).manual_seed(int(seed))
    for pair in range(view.num_pairs):
        layers = torch.nonzero(view.layer_present[pair], as_tuple=False).flatten()
        if len(layers) == 0:
            continue
        layers = layers[torch.randperm(len(layers), generator=generator, device=view.device)]
        for index, layer in enumerate(layers.tolist()):
            schedule[index % folds][pair, layer] = True
    return [mask for mask in schedule if bool(mask.any())]


@torch.no_grad()
def score_flat1024(model: Flat1024Model, view: FlatPairView, *, rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one local reconstruction feature per response token."""

    model.eval()
    pair_layer_error = view.value.new_full((view.num_pairs, view.num_layers), torch.nan)
    for block_mask in _scoring_schedule(view, rounds, seed):
        value, observed = apply_layer_block_mask(view, block_mask)
        prediction = model(view, value=value, observed=observed)
        pair_layer_error[block_mask] = block_vector_error(prediction[block_mask], view.value[block_mask])

    token_total = view.value.new_zeros(view.num_response_tokens)
    token_count = view.value.new_zeros(view.num_response_tokens)
    available = torch.isfinite(pair_layer_error) & view.layer_present
    if bool(available.any()):
        pair, _ = torch.nonzero(available, as_tuple=True)
        current = pair_layer_error[available]
        token = view.pair_query[pair]
        token_total.index_add_(0, token, current)
        token_count.index_add_(0, token, torch.ones_like(current))
    feature = view.value.new_full((view.num_response_tokens, 1), torch.nan)
    valid = token_count > 0
    feature[valid, 0] = token_total[valid] / token_count[valid]
    return feature.cpu().numpy().astype(np.float32), token_count[:, None].cpu().numpy().astype(np.float32)
