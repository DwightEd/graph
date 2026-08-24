"""Build a mass-conserving causal multiplex attention-provenance graph.

Only the public ``ResearchSample`` seam is used: ``attention()`` supplies
geometry/diagonal values and ``iter_sparse_attention_blocks()`` supplies exact
retained off-diagonal entries.  No cache implementation is imported here.

An edge is stored in information-flow direction ``source -> target``.  The
result remains an attention-provenance proxy: attention weights alone do not
contain value/output projections, residual mixing, or MLP computation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import GraphConfig


RP = 0
RR_NEAR = 1
RR_FAR = 2
RELATION_NAMES = ("prompt", "response_near", "response_far")
ROLE_NAMES = ("prompt", "response_history", "self", "unresolved")

# The canonical cache stores attention probabilities in float16.  Correctly
# normalized rows can therefore exceed one slightly after independent rounding,
# but an error materially larger than one float16 unit roundoff is not a
# numerical drift that this representation is allowed to hide.
MAX_NUMERICAL_OVERSHOOT = 1e-3


def _integer_dtype(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )


@dataclass(frozen=True)
class OvershootAudit:
    """Per-row record of the sole numerical correction applied by the builder."""

    raw_known_mass: torch.Tensor
    row_scale: torch.Tensor
    overshoot: torch.Tensor

    def validate(self, shape: tuple[int, int, int], device: torch.device) -> None:
        for name, tensor in (
            ("raw_known_mass", self.raw_known_mass),
            ("row_scale", self.row_scale),
            ("overshoot", self.overshoot),
        ):
            if tensor.shape != shape:
                raise ValueError(f"audit {name} must have shape {shape}")
            if tensor.device != device:
                raise ValueError("audit tensors must share the graph device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"audit {name} must be finite")
        if bool((self.raw_known_mass < 0).any()) or bool((self.overshoot < 0).any()):
            raise ValueError("audit masses must be non-negative")
        if bool(((self.row_scale <= 0) | (self.row_scale > 1)).any()):
            raise ValueError("row_scale must be in (0,1]")
        expected = (self.raw_known_mass - 1.0).clamp_min(0.0)
        if not torch.allclose(self.overshoot, expected, atol=2e-6, rtol=2e-6):
            raise ValueError("overshoot does not match raw_known_mass")
        if bool((self.overshoot > MAX_NUMERICAL_OVERSHOOT).any()):
            raise ValueError(
                "attention-row overshoot exceeds the permitted float16 "
                "rounding tolerance"
            )

    @property
    def num_overshoot_rows(self) -> int:
        return int((self.overshoot > 0).sum().item())

    @property
    def maximum_overshoot(self) -> float:
        if self.overshoot.numel() == 0:
            return 0.0
        return float(self.overshoot.max().item())


@dataclass(frozen=True)
class CausalRoutingGraph:
    """Exact retained endpoints plus four exhaustive masses for every row."""

    sample_id: str
    num_response_tokens: int
    num_layers: int
    num_heads: int
    num_tokens: int
    response_idx: int
    attention_floor: float
    recent_lag: int
    source: torch.Tensor
    target: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    relation: torch.Tensor
    raw_weight: torch.Tensor
    weight: torch.Tensor
    diagonal: torch.Tensor
    prompt_mass: torch.Tensor
    response_mass: torch.Tensor
    self_mass: torch.Tensor
    unresolved_mass: torch.Tensor
    audit: OvershootAudit

    @property
    def device(self) -> torch.device:
        return self.prompt_mass.device

    @property
    def num_channels(self) -> int:
        return int(self.num_layers) * int(self.num_heads)

    @property
    def num_edges(self) -> int:
        return int(self.weight.numel())

    @property
    def query(self) -> torch.Tensor:
        """Response-relative target token for every information-flow edge."""

        return self.target - int(self.response_idx)

    @property
    def edge_channel(self) -> torch.Tensor:
        return self.layer * int(self.num_heads) + self.head

    @property
    def role_mass(self) -> torch.Tensor:
        """Four exhaustive role masses in ``[R,L,H,4]`` form."""

        return torch.stack(
            (
                self.prompt_mass,
                self.response_mass,
                self.self_mass,
                self.unresolved_mass,
            ),
            dim=-1,
        )

    @property
    def channel_role_mass(self) -> torch.Tensor:
        """Flatten layer/head without averaging: ``[R,L*H,4]``."""

        return self.role_mass.reshape(self.num_response_tokens, self.num_channels, 4)

    @property
    def prompt_channel(self) -> torch.Tensor:
        return self.prompt_mass.reshape(self.num_response_tokens, self.num_channels)

    @property
    def response_channel(self) -> torch.Tensor:
        return self.response_mass.reshape(self.num_response_tokens, self.num_channels)

    @property
    def self_channel(self) -> torch.Tensor:
        return self.self_mass.reshape(self.num_response_tokens, self.num_channels)

    @property
    def unresolved_channel(self) -> torch.Tensor:
        return self.unresolved_mass.reshape(self.num_response_tokens, self.num_channels)

    def to(self, device: str | torch.device) -> "CausalRoutingGraph":
        audit = OvershootAudit(
            raw_known_mass=self.audit.raw_known_mass.to(device),
            row_scale=self.audit.row_scale.to(device),
            overshoot=self.audit.overshoot.to(device),
        )
        tensor_names = (
            "source",
            "target",
            "layer",
            "head",
            "relation",
            "raw_weight",
            "weight",
            "diagonal",
            "prompt_mass",
            "response_mass",
            "self_mass",
            "unresolved_mass",
        )
        return replace(
            self,
            **{name: getattr(self, name).to(device) for name in tensor_names},
            audit=audit,
        )

    def validate(self) -> "CausalRoutingGraph":
        if min(self.num_response_tokens, self.num_layers, self.num_heads) < 1:
            raise ValueError("graph response/layer/head geometry must be positive")
        if not 0 < self.response_idx < self.num_tokens:
            raise ValueError("response_idx must split a non-empty prompt and response")
        if self.num_tokens - self.response_idx != self.num_response_tokens:
            raise ValueError("response geometry is inconsistent")
        if self.recent_lag < 1:
            raise ValueError("recent_lag must be positive")
        if not 0.0 < float(self.attention_floor) <= 1.0:
            raise ValueError("attention_floor must be in (0,1]")

        edge_columns = (
            self.source,
            self.target,
            self.layer,
            self.head,
            self.relation,
            self.raw_weight,
            self.weight,
        )
        if any(tensor.ndim != 1 or tensor.numel() != self.num_edges for tensor in edge_columns):
            raise ValueError("edge arrays must be aligned one-dimensional tensors")
        if any(tensor.device != self.device for tensor in edge_columns):
            raise ValueError("all graph tensors must share one device")
        for tensor in (self.source, self.target, self.layer, self.head, self.relation):
            if not _integer_dtype(tensor):
                raise ValueError("edge index and relation arrays must be integer")
        if not bool(torch.isfinite(self.raw_weight).all()) or not bool(
            torch.isfinite(self.weight).all()
        ):
            raise ValueError("edge weights must be finite")
        if bool((self.raw_weight < 0).any()) or bool((self.weight < 0).any()):
            raise ValueError("edge weights must be non-negative")
        if bool((self.weight - self.raw_weight > 2e-6).any()):
            raise ValueError("numerical correction may only reduce edge weights")
        if self.num_edges:
            if bool(((self.target < self.response_idx) | (self.target >= self.num_tokens)).any()):
                raise ValueError("edge targets must be response tokens")
            if bool(((self.source < 0) | (self.source >= self.target)).any()):
                raise ValueError("edges must be strictly prefix-causal")
            if bool(((self.layer < 0) | (self.layer >= self.num_layers)).any()):
                raise ValueError("edge layer is out of range")
            if bool(((self.head < 0) | (self.head >= self.num_heads)).any()):
                raise ValueError("edge head is out of range")
            if bool(((self.relation < RP) | (self.relation > RR_FAR)).any()):
                raise ValueError("unsupported edge relation")
            prompt = self.source < self.response_idx
            lag = self.target - self.source
            expected_relation = torch.where(
                prompt,
                torch.full_like(self.relation, RP),
                torch.where(
                    lag <= self.recent_lag,
                    torch.full_like(self.relation, RR_NEAR),
                    torch.full_like(self.relation, RR_FAR),
                ),
            )
            if bool((self.relation != expected_relation).any()):
                raise ValueError("edge relation disagrees with its causal endpoint")

        shape = (self.num_response_tokens, self.num_layers, self.num_heads)
        mass_columns = (
            self.diagonal,
            self.prompt_mass,
            self.response_mass,
            self.self_mass,
            self.unresolved_mass,
        )
        if any(tensor.shape != shape for tensor in mass_columns):
            raise ValueError(f"node role tensors must have shape {shape}")
        if any(tensor.device != self.device for tensor in mass_columns):
            raise ValueError("node role tensors must share the graph device")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in mass_columns):
            raise ValueError("node role tensors must be finite")
        if any(bool((tensor < 0).any()) for tensor in mass_columns):
            raise ValueError("node role tensors must be non-negative")
        total = self.role_mass.sum(dim=-1)
        if not torch.allclose(total, torch.ones_like(total), atol=3e-5, rtol=3e-5):
            raise ValueError("role masses must conserve every attention row")

        # Verify that the corrected exact edges and the summarized RP/RR masses
        # are the same representation rather than two drifting copies.
        aggregate = torch.zeros(
            (self.num_response_tokens * self.num_channels, 2),
            dtype=self.weight.dtype,
            device=self.device,
        )
        if self.num_edges:
            flat_row = self.query * self.num_channels + self.edge_channel
            role = (self.relation != RP).long()
            aggregate.index_put_((flat_row, role), self.weight, accumulate=True)
        aggregate = aggregate.reshape(self.num_response_tokens, self.num_layers, self.num_heads, 2)
        if not torch.allclose(aggregate[..., 0], self.prompt_mass, atol=3e-5, rtol=3e-5):
            raise ValueError("prompt edge weights disagree with prompt_mass")
        if not torch.allclose(aggregate[..., 1], self.response_mass, atol=3e-5, rtol=3e-5):
            raise ValueError("RR edge weights disagree with response_mass")
        self.audit.validate(shape, self.device)
        expected_self = self.diagonal * self.audit.row_scale
        if not torch.allclose(expected_self, self.self_mass, atol=3e-5, rtol=3e-5):
            raise ValueError("self_mass does not match scaled exact diagonal")
        return self


def build_causal_routing_graph(
    sample,
    *,
    config: GraphConfig | None = None,
) -> CausalRoutingGraph:
    """Decode one public research sample without fabricating censored edges."""

    config = GraphConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_idx = int(attention.response_idx)
    response_count = int(attention.num_response_tokens)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    shape = (response_count, layers, heads)
    device = attention.response_values.device

    parts: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("source", "target", "layer", "head", "relation", "raw_weight")
    }
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        source = block.source.long()
        target = block.target.long()
        if source.numel() != target.numel():
            raise ValueError("sparse attention source/target columns are misaligned")
        if source.numel() == 0:
            continue
        if bool((source >= target).any()) or bool((source < 0).any()):
            raise ValueError("sparse attention edges must be strictly prefix-causal")
        if bool(((target < response_idx) | (target >= attention.num_tokens)).any()):
            raise ValueError("sparse attention targets must be response tokens")
        block_layer = block.layer.long()
        block_head = block.head.long()
        if block_layer.numel() != source.numel() or block_head.numel() != source.numel():
            raise ValueError("sparse attention channel columns are misaligned")
        if bool(((block_layer < 0) | (block_layer >= layers)).any()) or bool(
            ((block_head < 0) | (block_head >= heads)).any()
        ):
            raise ValueError("sparse attention channel index is out of range")
        raw_block_weight = block.weight.float()
        if raw_block_weight.numel() != source.numel():
            raise ValueError("sparse attention weights are misaligned")
        if not bool(torch.isfinite(raw_block_weight).all()) or bool(
            (raw_block_weight < 0).any()
        ):
            raise ValueError("sparse attention weights must be finite and non-negative")
        lag = target - source
        prompt = source < response_idx
        relation = torch.where(
            prompt,
            torch.full_like(source, RP),
            torch.where(
                lag <= config.recent_lag,
                torch.full_like(source, RR_NEAR),
                torch.full_like(source, RR_FAR),
            ),
        )
        parts["source"].append(source)
        parts["target"].append(target)
        parts["layer"].append(block_layer)
        parts["head"].append(block_head)
        parts["relation"].append(relation.long())
        parts["raw_weight"].append(raw_block_weight)

    if parts["source"]:
        source, target, layer, head, relation, raw_weight = (
            torch.cat(parts[name])
            for name in ("source", "target", "layer", "head", "relation", "raw_weight")
        )
    else:
        source = torch.empty(0, dtype=torch.long, device=device)
        target = torch.empty_like(source)
        layer = torch.empty_like(source)
        head = torch.empty_like(source)
        relation = torch.empty_like(source)
        raw_weight = torch.empty(0, dtype=torch.float32, device=device)

    diagonal = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    raw_prompt = torch.zeros(shape, dtype=torch.float32, device=device)
    raw_response = torch.zeros_like(raw_prompt)
    if raw_weight.numel():
        query = target - response_idx
        is_prompt = relation == RP
        if bool(is_prompt.any()):
            raw_prompt.index_put_(
                (query[is_prompt], layer[is_prompt], head[is_prompt]),
                raw_weight[is_prompt],
                accumulate=True,
            )
        is_response = ~is_prompt
        if bool(is_response.any()):
            raw_response.index_put_(
                (query[is_response], layer[is_response], head[is_response]),
                raw_weight[is_response],
                accumulate=True,
            )

    raw_known = raw_prompt + raw_response + diagonal
    overshoot = (raw_known - 1.0).clamp_min(0.0)
    if bool((overshoot > MAX_NUMERICAL_OVERSHOOT).any()):
        maximum = float(overshoot.max().item())
        raise ValueError(
            "retained attention plus diagonal exceeds one by more than the "
            f"float16 rounding tolerance: maximum overshoot={maximum:.8g}"
        )
    row_scale = torch.ones_like(raw_known)
    oversized = raw_known > 1.0
    row_scale[oversized] = raw_known[oversized].reciprocal()
    prompt_mass = raw_prompt * row_scale
    response_mass = raw_response * row_scale
    self_mass = diagonal * row_scale
    corrected_known = prompt_mass + response_mass + self_mass
    unresolved_mass = (1.0 - corrected_known).clamp_min(0.0)
    weight = raw_weight
    if raw_weight.numel():
        query = target - response_idx
        weight = raw_weight * row_scale[query, layer, head]

    graph = CausalRoutingGraph(
        sample_id=str(getattr(sample, "sample_id", "")),
        num_response_tokens=response_count,
        num_layers=layers,
        num_heads=heads,
        num_tokens=int(attention.num_tokens),
        response_idx=response_idx,
        attention_floor=float(attention.attention_floor),
        recent_lag=int(config.recent_lag),
        source=source,
        target=target,
        layer=layer,
        head=head,
        relation=relation,
        raw_weight=raw_weight,
        weight=weight,
        diagonal=diagonal,
        prompt_mass=prompt_mass,
        response_mass=response_mass,
        self_mass=self_mass,
        unresolved_mass=unresolved_mass,
        audit=OvershootAudit(
            raw_known_mass=raw_known,
            row_scale=row_scale,
            overshoot=overshoot,
        ),
    )
    return graph.validate()
