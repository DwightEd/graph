"""Construct pair-specific cross-head attention codes from a sparse token graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments.grounded_route.graph import TokenGraph

from .operators import OperatorGeometry


PAIR_RETAINED = 0
PAIR_SELF = 1


@dataclass(frozen=True)
class PairCodeField:
    """Sparse ``(layer, target, source)`` field with one vector over heads."""

    layer: torch.Tensor
    target: torch.Tensor
    source: torch.Tensor
    kind: torch.Tensor
    code: torch.Tensor
    observed: torch.Tensor
    magnitude: torch.Tensor
    direction: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.layer.numel())

    @property
    def head_count(self) -> int:
        return int(self.code.shape[1])

    def validate(self) -> "PairCodeField":
        count = self.count
        if any(
            value.shape != (count,)
            for value in (self.layer, self.target, self.source, self.kind, self.magnitude)
        ):
            raise ValueError("pair-code index vectors must align")
        if self.code.ndim != 2 or self.code.shape[0] != count:
            raise ValueError("pair code must be [pair, head]")
        if self.observed.shape != self.code.shape or self.direction.shape != self.code.shape:
            raise ValueError("pair-code tensors must align")
        if not torch.isfinite(self.code).all() or not torch.isfinite(self.direction).all():
            raise ValueError("pair-code field contains non-finite values")
        if bool((self.code < 0).any()) or bool((self.magnitude < 0).any()):
            raise ValueError("attention codes must be non-negative")
        return self

    def operator_embedding(
        self,
        geometry: OperatorGeometry,
        mode: str,
        *,
        seed: int = 0,
        use_direction: bool = True,
    ) -> torch.Tensor:
        """Embed codes so Euclidean distance matches the chosen operator metric."""

        if geometry.layer_count <= int(self.layer.max().item() if self.count else -1):
            raise ValueError("operator geometry has fewer layers than pair codes")
        if geometry.head_count != self.head_count:
            raise ValueError("operator geometry and pair codes use different head counts")
        factor = geometry.factor_for(mode, seed=seed).to(
            device=self.code.device,
            dtype=self.code.dtype,
        )
        value = self.direction if use_direction else self.code
        output = torch.empty_like(value)
        for layer in torch.unique(self.layer).tolist():
            selected = self.layer == int(layer)
            output[selected] = value[selected] @ factor[int(layer)]
        return output


def _impute_code(
    code: torch.Tensor,
    observed: torch.Tensor,
    *,
    attention_floor: float,
    mode: str,
    self_pair: bool,
) -> torch.Tensor:
    if self_pair:
        return code
    if mode == "zero":
        return code
    if mode == "floor":
        return torch.where(observed, code, code.new_full((), attention_floor))
    if mode == "midpoint":
        return torch.where(observed, code, code.new_full((), attention_floor * 0.5))
    if mode == "excess":
        return torch.where(
            observed,
            (code - attention_floor).clamp_min(0.0),
            torch.zeros_like(code),
        )
    raise ValueError("imputation must be zero, floor, midpoint, or excess")


def build_pair_code_field(
    graph: TokenGraph,
    *,
    imputation: str = "zero",
    include_self: bool = True,
) -> PairCodeField:
    """Group retained typed edges by ``(layer, target, source)``.

    The sparse cache only identifies pair components retained above its floor.
    ``observed`` therefore remains part of the output contract; zero/floor
    imputation must never be interpreted as an exact reconstruction of censored
    head components.
    """

    graph = graph.canonicalize()
    heads = graph.head_count
    token_count = graph.token_count
    edges = graph.edges

    if edges.count:
        key = (edges.layer * token_count + edges.target) * token_count + edges.source
        pair_key, inverse = torch.unique(key, sorted=True, return_inverse=True)
        pair_count = len(pair_key)
        code = torch.zeros((pair_count, heads), dtype=edges.weight.dtype)
        observed = torch.zeros((pair_count, heads), dtype=torch.bool)
        code.index_put_((inverse, edges.head), edges.weight, accumulate=True)
        observed[inverse, edges.head] = True
        source = pair_key.remainder(token_count)
        endpoint = torch.div(pair_key, token_count, rounding_mode="floor")
        target = endpoint.remainder(token_count)
        layer = torch.div(endpoint, token_count, rounding_mode="floor")
        magnitude = code.sum(dim=1) / float(heads)
        code = _impute_code(
            code,
            observed,
            attention_floor=graph.attention_floor,
            mode=imputation,
            self_pair=False,
        )
        kind = torch.full((pair_count,), PAIR_RETAINED, dtype=torch.long)
    else:
        layer = torch.empty(0, dtype=torch.long)
        target = torch.empty(0, dtype=torch.long)
        source = torch.empty(0, dtype=torch.long)
        kind = torch.empty(0, dtype=torch.long)
        code = torch.empty((0, heads), dtype=graph.diagonal.dtype)
        observed = torch.empty((0, heads), dtype=torch.bool)
        magnitude = torch.empty(0, dtype=graph.diagonal.dtype)

    if include_self:
        response = torch.arange(graph.response_count)
        self_layer = torch.arange(graph.layer_count).repeat_interleave(graph.response_count)
        self_target = (
            graph.response_start + response
        ).repeat(graph.layer_count)
        self_code = graph.diagonal.permute(1, 0, 2).reshape(-1, heads).cpu()
        self_observed = torch.ones_like(self_code, dtype=torch.bool)
        layer = torch.cat((layer.cpu(), self_layer))
        target = torch.cat((target.cpu(), self_target))
        source = torch.cat((source.cpu(), self_target.clone()))
        kind = torch.cat(
            (
                kind.cpu(),
                torch.full((len(self_target),), PAIR_SELF, dtype=torch.long),
            )
        )
        code = torch.cat((code.cpu(), self_code))
        observed = torch.cat((observed.cpu(), self_observed))
        magnitude = torch.cat(
            (magnitude.cpu(), self_code.sum(dim=1) / float(heads))
        )
    else:
        layer, target, source, kind = (
            layer.cpu(),
            target.cpu(),
            source.cpu(),
            kind.cpu(),
        )
        code, observed = code.cpu(), observed.cpu()
        magnitude = magnitude.cpu()

    total = code.sum(dim=1, keepdim=True)
    direction = torch.where(
        total > 1e-12,
        code / total.clamp_min(1e-12),
        torch.zeros_like(code),
    )
    order_key = ((layer * token_count + target) * token_count + source) * 2 + kind
    order = torch.argsort(order_key, stable=True)
    return PairCodeField(
        layer=layer[order].long(),
        target=target[order].long(),
        source=source[order].long(),
        kind=kind[order].long(),
        code=code[order].float(),
        observed=observed[order],
        magnitude=magnitude[order].float(),
        direction=direction[order].float(),
    ).validate()
