"""Causal source-state propagation and grounding-aware relational aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SourceReuseGraph
from .edge_refinement import PairEncoding
from .grounding_config import GroundingGraphConfig


class SetReadout(nn.Module):
    """Permutation-invariant readout with learned, mean, and max summaries."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.empty = nn.Parameter(torch.zeros(hidden_dim))
        self.gate = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        active = mass > 0
        if not bool(active.any()):
            return self.empty
        values = values[active]
        mass = mass[active]
        weight = F.softmax(
            self.gate(values).squeeze(-1) + mass.clamp_min(1e-8).log(),
            dim=0,
        )
        weighted = (weight[:, None] * values).sum(dim=0)
        return self.output(
            torch.cat((weighted, values.mean(0), values.max(0).values))
        )


@dataclass
class SourceStateBank:
    birth: torch.Tensor
    reuse: torch.Tensor
    cumulative_mass: torch.Tensor
    last_used: torch.Tensor

    def detach(self) -> "SourceStateBank":
        self.birth = self.birth.detach()
        self.reuse = self.reuse.detach()
        return self


class SourceStateEncoder(nn.Module):
    """Maintain prompt/source birth states and optional causal reuse memories."""

    def __init__(self, config: GroundingGraphConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.prompt_position = nn.Sequential(
            nn.Linear(2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.response_birth = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.reuse_input = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.reuse_update = nn.GRUCell(hidden, hidden)
        self.combine = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )

    def initialize(self, graph: SourceReuseGraph) -> SourceStateBank:
        hidden = self.config.hidden_dim
        state = graph.weight.new_zeros((graph.num_tokens, hidden))
        if graph.response_idx:
            position = torch.arange(graph.response_idx, device=graph.device).float()
            normalized = position / float(max(graph.response_idx - 1, 1))
            prompt_feature = torch.stack((normalized, 1.0 - normalized), dim=-1)
            prompt_state = self.prompt_position(prompt_feature)
            state = state.index_copy(
                0,
                torch.arange(graph.response_idx, device=graph.device),
                prompt_state,
            )
        return SourceStateBank(
            birth=state,
            reuse=state.new_zeros(state.shape),
            cumulative_mass=graph.weight.new_zeros(graph.num_tokens),
            last_used=torch.full(
                (graph.num_tokens,),
                -1,
                dtype=torch.long,
                device=graph.device,
            ),
        )

    def representations(
        self,
        bank: SourceStateBank,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        birth = bank.birth.index_select(0, sources)
        if not self.config.use_reuse_memory:
            return birth
        reuse = bank.reuse.index_select(0, sources)
        return self.combine(torch.cat((birth, reuse), dim=-1))

    def seed_response(
        self,
        bank: SourceStateBank,
        *,
        source_index: int,
        token_embedding: torch.Tensor,
    ) -> None:
        index = torch.tensor(
            [source_index],
            dtype=torch.long,
            device=token_embedding.device,
        )
        value = self.response_birth(token_embedding)[None]
        bank.birth = bank.birth.index_copy(0, index, value)

    def update_reuse(
        self,
        bank: SourceStateBank,
        *,
        token: int,
        sources: torch.Tensor,
        pair_embedding: torch.Tensor,
        pair_mass: torch.Tensor,
        token_embedding: torch.Tensor,
    ) -> None:
        if sources.numel() == 0:
            return
        previous = bank.reuse.index_select(0, sources)
        last = bank.last_used.index_select(0, sources)
        gap = torch.where(
            last < 0,
            torch.full_like(last, token + 1),
            token - last,
        ).float()
        numeric = torch.stack(
            (
                torch.log1p(gap),
                torch.log1p(bank.cumulative_mass.index_select(0, sources)),
                torch.full_like(gap, math.log1p(token + 1)),
            ),
            dim=-1,
        )
        repeated = token_embedding.expand(pair_embedding.shape[0], -1)
        update_input = self.reuse_input(
            torch.cat((pair_embedding, repeated, numeric), dim=-1)
        )
        updated = self.reuse_update(update_input, previous)
        bank.reuse = bank.reuse.index_copy(0, sources, updated)
        bank.cumulative_mass.index_add_(0, sources, pair_mass.detach())
        bank.last_used[sources] = token


class RelationalGraphEncoder(nn.Module):
    """Aggregate refined source messages under full or origin-specific views."""

    def __init__(self, config: GroundingGraphConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.config = config
        self.message = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.readout = SetReadout(hidden, config.dropout)
        self.controls = nn.Sequential(
            nn.Linear(4, hidden),
            nn.PReLU(),
            nn.Linear(hidden, hidden),
        )
        self.token = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )

    def _controls(
        self,
        graph: SourceReuseGraph,
        *,
        token: int,
        pair_mass: torch.Tensor,
    ) -> torch.Tensor:
        channel_count = float(graph.num_layers * graph.num_heads)
        diagonal = graph.diagonal[token].mean()
        retained = pair_mass.sum() / channel_count
        unresolved = 1.0 - (diagonal + retained).clamp(0.0, 1.0)
        position = diagonal.new_tensor(math.log1p(token + 1) / 8.0)
        return self.controls(
            torch.stack((diagonal, retained, unresolved, position))
        )

    def forward(
        self,
        graph: SourceReuseGraph,
        *,
        token: int,
        pair: PairEncoding,
        source_state: torch.Tensor,
        view: str = "full",
        pair_multiplier: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pair.sources.numel() == 0:
            context = self.readout(pair.embedding, pair.mass)
            controls = self._controls(graph, token=token, pair_mass=pair.mass)
            return self.token(torch.cat((context, controls)))

        relation = (pair.sources >= graph.response_idx).float()
        message = self.message(
            torch.cat(
                (
                    pair.embedding,
                    source_state,
                    pair.origin[:, None],
                    relation[:, None],
                ),
                dim=-1,
            )
        )
        if view == "full":
            origin_factor = torch.ones_like(pair.origin)
        elif view == "no_prompt":
            origin_factor = 1.0 - pair.origin
        elif view == "no_response":
            origin_factor = pair.origin
        else:
            raise ValueError(f"unknown counterfactual view: {view}")
        if pair_multiplier is not None:
            origin_factor = origin_factor * pair_multiplier
        mass = pair.mass * pair.gate * origin_factor
        context = self.readout(message, mass)
        controls = self._controls(graph, token=token, pair_mass=pair.mass)
        return self.token(torch.cat((context, controls)))


class StructuredDecoder(nn.Module):
    """Decode received-support, grounding, and provenance fields."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: GroundingGraphConfig,
    ):
        super().__init__()
        hidden = config.hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.topk = config.received_topk
        self.received = nn.Linear(
            hidden,
            num_layers * num_heads * self.topk,
        )
        self.grounding = nn.Linear(hidden, num_layers * num_heads * 3)
        self.provenance = nn.Linear(hidden, num_layers)

    def forward(
        self,
        token_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        received = torch.sigmoid(self.received(token_embedding)).reshape(
            self.num_layers,
            self.num_heads,
            self.topk,
        )
        grounding = torch.sigmoid(self.grounding(token_embedding)).reshape(
            self.num_layers,
            self.num_heads,
            3,
        )
        provenance = torch.sigmoid(self.provenance(token_embedding))
        return received, grounding, provenance
