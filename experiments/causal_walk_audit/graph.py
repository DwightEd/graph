"""Mass-conserving causal attention graph with typed response edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GraphConfig

PROMPT = 0
RESPONSE_NEAR = 1
RESPONSE_FAR = 2
RELATION_NAMES = ("prompt", "response_near", "response_far")


@dataclass(frozen=True)
class RoutingGraph:
    sample_id: str
    response_idx: int
    num_response_tokens: int
    num_tokens: int
    num_layers: int
    num_heads: int
    attention_floor: float
    recent_lag: int

    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    relation: torch.Tensor
    weight: torch.Tensor

    prompt_mass: torch.Tensor
    response_mass: torch.Tensor
    self_mass: torch.Tensor
    unresolved_mass: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.prompt_mass.device

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    @property
    def query(self) -> torch.Tensor:
        return self.target - self.response_idx

    @property
    def channel(self) -> torch.Tensor:
        return self.layer * self.num_heads + self.head

    @property
    def role_mass(self) -> torch.Tensor:
        return torch.stack(
            (
                self.prompt_mass,
                self.response_mass,
                self.self_mass,
                self.unresolved_mass,
            ),
            dim=-1,
        )

    def validate(self) -> "RoutingGraph":
        if self.weight.numel():
            if bool((self.source >= self.target).any()):
                raise ValueError("attention graph must remain strictly prefix-causal")
            if bool(((self.layer < 0) | (self.layer >= self.num_layers)).any()):
                raise ValueError("layer index is outside the model geometry")
            if bool(((self.head < 0) | (self.head >= self.num_heads)).any()):
                raise ValueError("head index is outside the model geometry")
        total = self.role_mass.sum(dim=-1)
        if not torch.allclose(total, torch.ones_like(total), atol=3e-5, rtol=3e-5):
            raise ValueError("prompt/response/self/unresolved mass must sum to one")
        return self


@torch.no_grad()
def build_routing_graph(
    sample,
    *,
    config: GraphConfig | None = None,
) -> RoutingGraph:
    """Decode retained causal edges without fabricating censored values."""

    config = GraphConfig() if config is None else config
    attention = sample.attention()
    response_idx = int(attention.response_idx)
    response_tokens = int(attention.num_response_tokens)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    num_tokens = int(attention.num_tokens)
    device = attention.response_values.device

    columns: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("source", "target", "layer", "head", "relation", "weight")
    }
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        causal = block.source < block.target
        if not bool(causal.any()):
            continue
        source = block.source[causal].long()
        target = block.target[causal].long()
        layer = block.layer[causal].long()
        head = block.head[causal].long()
        weight = block.weight[causal].float().clamp_min(0.0)
        lag = target - source
        prompt = source < response_idx
        relation = torch.where(
            prompt,
            torch.full_like(source, PROMPT),
            torch.where(
                lag <= config.recent_lag,
                torch.full_like(source, RESPONSE_NEAR),
                torch.full_like(source, RESPONSE_FAR),
            ),
        )
        for name, value in (
            ("source", source),
            ("target", target),
            ("layer", layer),
            ("head", head),
            ("relation", relation),
            ("weight", weight),
        ):
            columns[name].append(value)

    if columns["source"]:
        source, target, layer, head, relation, raw_weight = (
            torch.cat(columns[name])
            for name in ("source", "target", "layer", "head", "relation", "weight")
        )
    else:
        source = torch.empty(0, dtype=torch.long, device=device)
        target = torch.empty_like(source)
        layer = torch.empty_like(source)
        head = torch.empty_like(source)
        relation = torch.empty_like(source)
        raw_weight = torch.empty(0, dtype=torch.float32, device=device)

    shape = (response_tokens, layers, heads)
    prompt_mass = torch.zeros(shape, dtype=torch.float32, device=device)
    response_mass = torch.zeros_like(prompt_mass)
    if raw_weight.numel():
        query = target - response_idx
        is_prompt = relation == PROMPT
        if bool(is_prompt.any()):
            prompt_mass.index_put_(
                (query[is_prompt], layer[is_prompt], head[is_prompt]),
                raw_weight[is_prompt],
                accumulate=True,
            )
        is_response = ~is_prompt
        if bool(is_response.any()):
            response_mass.index_put_(
                (query[is_response], layer[is_response], head[is_response]),
                raw_weight[is_response],
                accumulate=True,
            )

    self_mass = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    known = prompt_mass + response_mass + self_mass
    overshoot = (known - 1.0).clamp_min(0.0)
    if overshoot.numel() and float(overshoot.max().item()) > config.numerical_tolerance:
        raise ValueError("attention row mass exceeds the numerical tolerance")

    scale = torch.where(known > 1.0, known.reciprocal(), torch.ones_like(known))
    prompt_mass = prompt_mass * scale
    response_mass = response_mass * scale
    self_mass = self_mass * scale
    if raw_weight.numel():
        raw_weight = raw_weight * scale[target - response_idx, layer, head]
    unresolved = (1.0 - prompt_mass - response_mass - self_mass).clamp_min(0.0)

    return RoutingGraph(
        sample_id=str(sample.sample_id),
        response_idx=response_idx,
        num_response_tokens=response_tokens,
        num_tokens=num_tokens,
        num_layers=layers,
        num_heads=heads,
        attention_floor=float(attention.attention_floor),
        recent_lag=config.recent_lag,
        source=source,
        target=target,
        layer=layer,
        head=head,
        relation=relation,
        weight=raw_weight,
        prompt_mass=prompt_mass,
        response_mass=response_mass,
        self_mass=self_mass,
        unresolved_mass=unresolved,
    ).validate()
