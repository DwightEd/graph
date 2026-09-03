"""Exact functional-message graph construction.

The dense node profile uses every causal source.  The explicit edge list is a
serialization view of the largest functional messages; the omitted mass is
kept exactly in ``edge_tail_profile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

ROLE_NAMES = ("evidence", "other_prompt", "response_history", "predictor_self")
PROFILE_CHANNELS = (
    "attention",
    "residual_message_norm",
    "positive_function",
    "negative_function",
)
STAGE_PRE, STAGE_ATTN, STAGE_OUT = range(3)
EDGE_ATTENTION, EDGE_RESIDUAL, EDGE_MLP, EDGE_LAYER, EDGE_OUTPUT = range(5)


def source_roles(
    source_count: int,
    response_start: int,
    predictor: int,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Return one structural role for every causal source position."""

    source = torch.arange(source_count, device=evidence_mask.device)
    role = torch.full_like(source, 1, dtype=torch.int64)
    prompt = source < response_start
    evidence = torch.zeros(source_count, dtype=torch.bool, device=source.device)
    evidence[: min(response_start, source_count)] = evidence_mask[:source_count]
    role[prompt & evidence] = 0
    role[(source >= response_start) & (source < predictor)] = 2
    role[source == predictor] = 3
    return role


def _profile(
    attention: torch.Tensor,
    transport: torch.Tensor,
    function: torch.Tensor,
    roles: torch.Tensor,
) -> torch.Tensor:
    """Aggregate the full head-source tensor without averaging heads or layers."""

    result = attention.new_zeros((attention.shape[0], len(ROLE_NAMES), 4))
    for role in range(len(ROLE_NAMES)):
        mask = roles == role
        if not bool(mask.any()):
            continue
        result[:, role, 0] = attention[:, mask].sum(-1)
        result[:, role, 1] = transport[:, mask].sum(-1)
        selected = function[:, mask]
        result[:, role, 2] = selected.clamp_min(0).sum(-1)
        result[:, role, 3] = (-selected).clamp_min(0).sum(-1)
    return result


def _keep_indices(priority: torch.Tensor, cover: float, budget: int) -> torch.Tensor:
    """Choose a compact explicit view after the dense profile has been computed."""

    flat = priority.flatten()
    if budget == 0 or budget >= flat.numel():
        return torch.arange(flat.numel(), device=flat.device)
    values, order = flat.topk(budget)
    total = flat.sum()
    if float(total) <= 0:
        return order[:0]
    required = int(torch.searchsorted(values.cumsum(0), cover * total).item()) + 1
    return order[: min(required, budget)]


@dataclass
class FunctionalMessageGraph:
    token_ids: torch.Tensor
    response_start: int
    node_profile: torch.Tensor
    mlp_profile: torch.Tensor
    node_embedding: torch.Tensor
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_head: torch.Tensor
    edge_role: torch.Tensor
    edge_response: torch.Tensor
    edge_attention: torch.Tensor
    edge_function: torch.Tensor
    edge_residual_norm: torch.Tensor
    edge_head_message: torch.Tensor
    edge_tail_profile: torch.Tensor
    structure_edge_index: torch.Tensor
    structure_edge_type: torch.Tensor
    output_node: torch.Tensor
    target_logprob: torch.Tensor
    target_margin: torch.Tensor


@dataclass
class GraphBuilder:
    token_ids: torch.Tensor
    response_start: int
    layers: int
    heads: int
    head_dim: int
    edge_cover: float = 0.95
    edge_budget: int = 64
    profile: torch.Tensor = field(init=False)
    selected_profile: torch.Tensor = field(init=False)
    mlp_profile: torch.Tensor = field(init=False)
    target_logprob: torch.Tensor = field(init=False)
    target_margin: torch.Tensor = field(init=False)
    _source: list[torch.Tensor] = field(default_factory=list)
    _target: list[torch.Tensor] = field(default_factory=list)
    _layer: list[torch.Tensor] = field(default_factory=list)
    _head: list[torch.Tensor] = field(default_factory=list)
    _role: list[torch.Tensor] = field(default_factory=list)
    _response: list[torch.Tensor] = field(default_factory=list)
    _attention: list[torch.Tensor] = field(default_factory=list)
    _function: list[torch.Tensor] = field(default_factory=list)
    _residual_norm: list[torch.Tensor] = field(default_factory=list)
    _message: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self) -> None:
        response = len(self.token_ids) - self.response_start
        shape = (response, self.layers, self.heads, len(ROLE_NAMES), len(PROFILE_CHANNELS))
        self.profile = torch.zeros(shape, dtype=torch.float32)
        self.selected_profile = torch.zeros_like(self.profile)
        self.mlp_profile = torch.zeros(response, self.layers, 3, dtype=torch.float32)
        self.target_logprob = torch.zeros(response, dtype=torch.float32)
        self.target_margin = torch.zeros(response, dtype=torch.float32)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def state_node(
        self, layer: int, stage: int, token: torch.Tensor | int
    ) -> torch.Tensor:
        offset = (layer * 3 + stage) * self.token_count
        return torch.as_tensor(offset) + torch.as_tensor(token, device="cpu")

    def add_layer(
        self,
        *,
        target: int,
        predictor: int,
        layer: int,
        attention: torch.Tensor,
        value: torch.Tensor,
        head_gradient: torch.Tensor,
        output_gram: torch.Tensor,
        q_to_kv: torch.Tensor,
        roles: torch.Tensor,
        mlp_output: torch.Tensor,
        mlp_gradient: torch.Tensor,
    ) -> None:
        """Add one layer using exact ``A·V·W_O`` gradient contributions."""

        attention = attention.detach().float()
        value = value.detach()[:, q_to_kv].float()
        head_gradient = head_gradient.detach().float().reshape(
            self.heads, self.head_dim
        )
        output_gram = output_gram.detach().float()
        head_message = attention.T[..., None] * value
        function = attention * torch.einsum("shd,hd->hs", value, head_gradient)
        transport = torch.einsum(
            "shd,hde,she->hs", head_message, output_gram, head_message
        ).clamp_min(0).sqrt()
        dense = _profile(attention, transport, function, roles)
        self.profile[target, layer] = dense.cpu()

        priority = function.abs()
        if float(priority.sum()) == 0:
            priority = transport
        kept = _keep_indices(priority, self.edge_cover, self.edge_budget)
        head = torch.div(kept, attention.shape[1], rounding_mode="floor")
        source = kept.remainder(attention.shape[1])
        message = head_message[source, head]

        kept_profile = _profile(
            attention.new_zeros(attention.shape).index_put((head, source), attention[head, source]),
            transport.new_zeros(transport.shape).index_put((head, source), transport[head, source]),
            function.new_zeros(function.shape).index_put((head, source), function[head, source]),
            roles,
        )
        self.selected_profile[target, layer] = kept_profile.cpu()

        self._source.append(self.state_node(layer, STAGE_PRE, source.cpu()))
        self._target.append(
            self.state_node(layer, STAGE_ATTN, predictor).expand(len(source))
        )
        self._layer.append(torch.full((len(source),), layer, dtype=torch.int16))
        self._head.append(head.cpu().to(torch.int16))
        self._role.append(roles[source].cpu().to(torch.int8))
        self._response.append(torch.full((len(source),), target, dtype=torch.int32))
        self._attention.append(attention[head, source].detach().cpu().to(torch.float16))
        self._function.append(function[head, source].cpu())
        self._residual_norm.append(transport[head, source].cpu().to(torch.float16))
        self._message.append(message.cpu().to(torch.float16))

        mlp_function = (mlp_output.float() * mlp_gradient.float()).sum()
        self.mlp_profile[target, layer] = torch.tensor(
            [
                mlp_output.float().norm().item(),
                mlp_function.clamp_min(0).item(),
                (-mlp_function).clamp_min(0).item(),
            ]
        )

    def add_target_score(self, target: int, logprob: torch.Tensor, margin: torch.Tensor) -> None:
        self.target_logprob[target] = logprob.detach().float().cpu()
        self.target_margin[target] = margin.detach().float().cpu()

    def _structure(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source, target, kind = [], [], []
        response = len(self.token_ids) - self.response_start
        output_offset = 3 * self.layers * self.token_count
        output_node = torch.arange(response, dtype=torch.int64) + output_offset
        for t in range(response):
            predictor = self.response_start - 1 + t
            for layer in range(self.layers):
                pre = int(self.state_node(layer, STAGE_PRE, predictor))
                attn = int(self.state_node(layer, STAGE_ATTN, predictor))
                out = int(self.state_node(layer, STAGE_OUT, predictor))
                source.extend((pre, attn, attn))
                target.extend((attn, out, out))
                kind.extend((EDGE_RESIDUAL, EDGE_RESIDUAL, EDGE_MLP))
                if layer + 1 < self.layers:
                    source.append(out)
                    target.append(int(self.state_node(layer + 1, STAGE_PRE, predictor)))
                    kind.append(EDGE_LAYER)
            source.append(int(self.state_node(self.layers - 1, STAGE_OUT, predictor)))
            target.append(int(output_node[t]))
            kind.append(EDGE_OUTPUT)
        return (
            torch.tensor((source, target), dtype=torch.int64),
            torch.tensor(kind, dtype=torch.int8),
            output_node,
        )

    def finish(self) -> FunctionalMessageGraph:
        structure_index, structure_type, output_node = self._structure()
        if self._source:
            edge_index = torch.stack((torch.cat(self._source), torch.cat(self._target)))
            edge_layer = torch.cat(self._layer)
            edge_head = torch.cat(self._head)
            edge_role = torch.cat(self._role)
            edge_response = torch.cat(self._response)
            edge_attention = torch.cat(self._attention)
            edge_function = torch.cat(self._function)
            edge_residual_norm = torch.cat(self._residual_norm)
            edge_message = torch.cat(self._message)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.int64)
            edge_layer = torch.empty(0, dtype=torch.int16)
            edge_head = torch.empty(0, dtype=torch.int16)
            edge_role = torch.empty(0, dtype=torch.int8)
            edge_response = torch.empty(0, dtype=torch.int32)
            edge_attention = torch.empty(0, dtype=torch.float16)
            edge_function = torch.empty(0, dtype=torch.float32)
            edge_residual_norm = torch.empty(0, dtype=torch.float16)
            edge_message = torch.empty(0, self.head_dim, dtype=torch.float16)

        profile = self.profile.to(torch.float16)
        mlp = self.mlp_profile.to(torch.float16)
        embedding = torch.cat((profile.flatten(1), mlp.flatten(1)), dim=1)
        return FunctionalMessageGraph(
            token_ids=self.token_ids.cpu(),
            response_start=self.response_start,
            node_profile=profile,
            mlp_profile=mlp,
            node_embedding=embedding,
            edge_index=edge_index,
            edge_layer=edge_layer,
            edge_head=edge_head,
            edge_role=edge_role,
            edge_response=edge_response,
            edge_attention=edge_attention,
            edge_function=edge_function,
            edge_residual_norm=edge_residual_norm,
            edge_head_message=edge_message,
            edge_tail_profile=(self.profile - self.selected_profile).to(torch.float16),
            structure_edge_index=structure_index,
            structure_edge_type=structure_type,
            output_node=output_node,
            target_logprob=self.target_logprob,
            target_margin=self.target_margin,
        )
