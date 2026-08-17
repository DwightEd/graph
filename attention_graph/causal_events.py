"""Reusable causal multiplex attention events.

This module is shared infrastructure for attention-graph experiments. It only
consumes ``ResearchSample`` views, preserves exact response source/target
identity and layer/head labels, and never fabricates prompt-token semantic
alignment. Missing CSR entries remain censored and are not emitted as observed
zero-weight edges.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch


RP = 0
RR = 1

SUMMARY_NAMES = (
    "rp_mass",
    "rr_mass",
    "rp_edge_count",
    "rr_edge_count",
    "rp_weight_log_weight",
    "rr_weight_log_weight",
    "rp_max_weight",
    "rr_max_weight",
    "all_weight_log_weight",
    "all_max_weight",
)


@dataclass(frozen=True)
class MultiplexEventConfig:
    """Streaming and balanced-selection controls."""

    block_rows: int = 8192
    layer_bands: int = 8
    max_prompt_events_per_band: int = 4
    max_rr_events_per_band: int = 8
    epsilon: float = 1e-8

    def validate(self) -> None:
        if int(self.block_rows) < 1:
            raise ValueError("block_rows must be positive")
        if int(self.layer_bands) < 1:
            raise ValueError("layer_bands must be positive")
        if int(self.max_prompt_events_per_band) < 0:
            raise ValueError("max_prompt_events_per_band must be non-negative")
        if int(self.max_rr_events_per_band) < 1:
            raise ValueError("max_rr_events_per_band must be positive")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class CausalMultiplexEvents:
    """Selected events plus full role/band summaries for one response.

    ``source`` is response-relative for RR events and ``-1`` for RP events.
    ``target_ptr`` partitions the selected event list by response target.
    ``role_summary`` and ``band_summary`` are computed from every retained event
    before top-k selection.
    """

    sample_id: str
    response_count: int
    num_layers: int
    num_heads: int
    attention_floor: float
    layer_bands: int
    target_ptr: torch.Tensor
    relation: torch.Tensor
    source: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    weight: torch.Tensor
    lag: torch.Tensor
    role_summary: torch.Tensor
    band_summary: torch.Tensor

    @property
    def num_channels(self) -> int:
        return int(self.num_layers) * int(self.num_heads)

    @property
    def num_events(self) -> int:
        return int(self.weight.numel())

    @property
    def channel(self) -> torch.Tensor:
        return self.layer * int(self.num_heads) + self.head

    @property
    def band(self) -> torch.Tensor:
        return torch.div(
            self.layer * int(self.layer_bands),
            int(self.num_layers),
            rounding_mode="floor",
        ).clamp_max(int(self.layer_bands) - 1)

    def target_slice(self, token_index: int) -> slice:
        token_index = int(token_index)
        if not 0 <= token_index < self.response_count:
            raise IndexError("response token is outside the event sample")
        return slice(
            int(self.target_ptr[token_index].item()),
            int(self.target_ptr[token_index + 1].item()),
        )

    def target_index(self) -> torch.Tensor:
        counts = self.target_ptr[1:] - self.target_ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(
                self.response_count,
                dtype=torch.long,
                device=self.target_ptr.device,
            ),
            counts,
        )

    def to(self, device) -> "CausalMultiplexEvents":
        return replace(
            self,
            target_ptr=self.target_ptr.to(device),
            relation=self.relation.to(device),
            source=self.source.to(device),
            layer=self.layer.to(device),
            head=self.head.to(device),
            weight=self.weight.to(device),
            lag=self.lag.to(device),
            role_summary=self.role_summary.to(device),
            band_summary=self.band_summary.to(device),
        )

    def validate(self) -> "CausalMultiplexEvents":
        if self.response_count < 1:
            raise ValueError("response_count must be positive")
        if self.num_layers < 1 or self.num_heads < 1 or self.layer_bands < 1:
            raise ValueError("attention geometry must be positive")
        if self.target_ptr.shape != (self.response_count + 1,):
            raise ValueError("target_ptr has the wrong shape")
        if int(self.target_ptr[0]) != 0 or int(self.target_ptr[-1]) != self.num_events:
            raise ValueError("target_ptr does not cover every selected event")
        if bool((self.target_ptr[1:] < self.target_ptr[:-1]).any()):
            raise ValueError("target_ptr must be non-decreasing")
        columns = (
            self.relation,
            self.source,
            self.layer,
            self.head,
            self.weight,
            self.lag,
        )
        if any(tensor.ndim != 1 or len(tensor) != self.num_events for tensor in columns):
            raise ValueError("event columns have inconsistent shapes")
        if self.role_summary.shape != (self.response_count, len(SUMMARY_NAMES)):
            raise ValueError("role_summary has the wrong shape")
        if self.band_summary.shape != (
            self.response_count,
            self.layer_bands,
            len(SUMMARY_NAMES),
        ):
            raise ValueError("band_summary has the wrong shape")
        if bool(((self.relation != RP) & (self.relation != RR)).any()):
            raise ValueError("unsupported event relation")
        if bool((self.layer < 0).any()) or bool((self.layer >= self.num_layers).any()):
            raise ValueError("event layer is outside the model geometry")
        if bool((self.head < 0).any()) or bool((self.head >= self.num_heads).any()):
            raise ValueError("event head is outside the model geometry")
        prompt = self.relation == RP
        response = self.relation == RR
        if bool((self.source[prompt] != -1).any()) or bool((self.lag[prompt] != 0).any()):
            raise ValueError("prompt events must use source -1 and zero lag")
        if bool((self.source[response] < 0).any()) or bool((self.lag[response] <= 0).any()):
            raise ValueError("RR events must preserve a legal source and lag")
        targets = self.target_index()
        if bool((self.source[response] >= targets[response]).any()):
            raise ValueError("RR event is not strictly causal")
        if not bool(torch.isfinite(self.weight).all()) or bool((self.weight < 0).any()):
            raise ValueError("event weights must be finite and non-negative")
        if not bool(torch.isfinite(self.role_summary).all()):
            raise ValueError("role summaries contain non-finite values")
        if not bool(torch.isfinite(self.band_summary).all()):
            raise ValueError("band summaries contain non-finite values")
        nonnegative_columns = (0, 1, 2, 3, 6, 7, 9)
        if bool((self.role_summary[:, nonnegative_columns] < 0).any()):
            raise ValueError("role mass/count/max summaries must be non-negative")
        if bool((self.band_summary[..., nonnegative_columns] < 0).any()):
            raise ValueError("band mass/count/max summaries must be non-negative")
        return self


def log_lag_bin(lag: int) -> int:
    lag = int(lag)
    if lag < 1:
        raise ValueError("causal lag must be positive")
    return int(np.floor(np.log2(lag)))


def _stable_top_indices(
    indices: torch.Tensor,
    weight: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Select strongest events with canonical CSR order as the tie break."""
    limit = int(limit)
    if limit <= 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)
    order = torch.argsort(weight[indices], descending=True, stable=True)
    return indices[order[:limit]]


def _scatter_max(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> None:
    if index.numel():
        target.scatter_reduce_(0, index, values, reduce="amax", include_self=True)


def _summaries(
    target: torch.Tensor,
    band: torch.Tensor,
    relation: torch.Tensor,
    weight: torch.Tensor,
    *,
    response_count: int,
    layer_bands: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact retained role summaries before event selection."""
    device = weight.device
    summary = torch.zeros(
        (response_count, len(SUMMARY_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    flat_summary = torch.zeros(
        (response_count * layer_bands, len(SUMMARY_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    flat_band = target * int(layer_bands) + band
    safe = weight.clamp_min(float(epsilon))
    wlogw = weight * torch.log(safe)
    for relation_value, mass_col, count_col, wlogw_col, max_col in (
        (RP, 0, 2, 4, 6),
        (RR, 1, 3, 5, 7),
    ):
        selected = relation == relation_value
        if not bool(selected.any()):
            continue
        row = target[selected]
        band_row = flat_band[selected]
        value = weight[selected]
        value_wlogw = wlogw[selected]
        one = torch.ones_like(value)
        summary[:, mass_col].index_add_(0, row, value)
        summary[:, count_col].index_add_(0, row, one)
        summary[:, wlogw_col].index_add_(0, row, value_wlogw)
        _scatter_max(summary[:, max_col], row, value)
        flat_summary[:, mass_col].index_add_(0, band_row, value)
        flat_summary[:, count_col].index_add_(0, band_row, one)
        flat_summary[:, wlogw_col].index_add_(0, band_row, value_wlogw)
        _scatter_max(flat_summary[:, max_col], band_row, value)
    summary[:, 8] = summary[:, 4] + summary[:, 5]
    summary[:, 9] = torch.maximum(summary[:, 6], summary[:, 7])
    flat_summary[:, 8] = flat_summary[:, 4] + flat_summary[:, 5]
    flat_summary[:, 9] = torch.maximum(flat_summary[:, 6], flat_summary[:, 7])
    return summary, flat_summary.reshape(
        response_count, layer_bands, len(SUMMARY_NAMES)
    )


def extract_causal_multiplex_events(
    sample,
    *,
    config: MultiplexEventConfig | None = None,
) -> CausalMultiplexEvents:
    """Build deterministic band-balanced events from a canonical sample."""
    config = MultiplexEventConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("events require a non-empty prompt and response")
    device = attention.response_values.device

    target_parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    layer_parts: list[torch.Tensor] = []
    head_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    lag_parts: list[torch.Tensor] = []
    band_parts: list[torch.Tensor] = []

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        target = block.query.long()
        layer = block.layer.long()
        head = block.head.long()
        weight = block.weight.float().clamp_min(0.0)
        prompt = block.source < prompt_count
        relation = torch.where(
            prompt,
            torch.full_like(target, RP),
            torch.full_like(target, RR),
        )
        source = torch.where(
            prompt,
            torch.full_like(block.source.long(), -1),
            block.source.long() - prompt_count,
        )
        lag = torch.where(prompt, torch.zeros_like(target), target - source)
        if bool((lag[~prompt] <= 0).any()):
            raise ValueError("canonical attention contains a non-causal RR edge")
        band = torch.div(
            layer * int(config.layer_bands),
            layers,
            rounding_mode="floor",
        ).clamp_max(int(config.layer_bands) - 1)

        target_parts.append(target)
        relation_parts.append(relation)
        source_parts.append(source.long())
        layer_parts.append(layer)
        head_parts.append(head)
        weight_parts.append(weight)
        lag_parts.append(lag.long())
        band_parts.append(band.long())

    if target_parts:
        target_all = torch.cat(target_parts)
        relation_all = torch.cat(relation_parts)
        source_all = torch.cat(source_parts)
        layer_all = torch.cat(layer_parts)
        head_all = torch.cat(head_parts)
        weight_all = torch.cat(weight_parts)
        lag_all = torch.cat(lag_parts)
        band_all = torch.cat(band_parts)
        order = torch.argsort(target_all, stable=True)
        count_by_target = torch.bincount(target_all, minlength=response_count)
        grouped_ptr = torch.zeros(
            response_count + 1, dtype=torch.long, device=device
        )
        grouped_ptr[1:] = torch.cumsum(count_by_target, dim=0)
    else:
        target_all = torch.empty(0, dtype=torch.long, device=device)
        relation_all = torch.empty_like(target_all)
        source_all = torch.empty_like(target_all)
        layer_all = torch.empty_like(target_all)
        head_all = torch.empty_like(target_all)
        weight_all = torch.empty(0, dtype=torch.float32, device=device)
        lag_all = torch.empty_like(target_all)
        band_all = torch.empty_like(target_all)
        order = torch.empty_like(target_all)
        grouped_ptr = torch.zeros(
            response_count + 1, dtype=torch.long, device=device
        )

    role_summary, band_summary = _summaries(
        target_all,
        band_all,
        relation_all,
        weight_all,
        response_count=response_count,
        layer_bands=config.layer_bands,
        epsilon=config.epsilon,
    )

    selected_parts: list[torch.Tensor] = []
    selected_count = torch.zeros(
        response_count, dtype=torch.long, device=device
    )
    for token in range(response_count):
        current = order[
            int(grouped_ptr[token].item()) : int(grouped_ptr[token + 1].item())
        ]
        current_parts: list[torch.Tensor] = []
        for band_index in range(config.layer_bands):
            in_band = current[band_all[current] == band_index]
            prompt_indices = in_band[relation_all[in_band] == RP]
            rr_indices = in_band[relation_all[in_band] == RR]
            current_parts.extend(
                (
                    _stable_top_indices(
                        prompt_indices,
                        weight_all,
                        config.max_prompt_events_per_band,
                    ),
                    _stable_top_indices(
                        rr_indices,
                        weight_all,
                        config.max_rr_events_per_band,
                    ),
                )
            )
        selected = (
            torch.cat(current_parts)
            if current_parts
            else torch.empty(0, dtype=torch.long, device=device)
        )
        selected_parts.append(selected)
        selected_count[token] = len(selected)

    selected = (
        torch.cat(selected_parts)
        if selected_parts
        else torch.empty(0, dtype=torch.long, device=device)
    )
    target_ptr = torch.zeros(
        response_count + 1, dtype=torch.long, device=device
    )
    target_ptr[1:] = torch.cumsum(selected_count, dim=0)

    return CausalMultiplexEvents(
        sample_id=str(sample.sample_id),
        response_count=response_count,
        num_layers=layers,
        num_heads=heads,
        attention_floor=float(attention.attention_floor),
        layer_bands=int(config.layer_bands),
        target_ptr=target_ptr,
        relation=relation_all[selected],
        source=source_all[selected],
        layer=layer_all[selected],
        head=head_all[selected],
        weight=weight_all[selected],
        lag=lag_all[selected],
        role_summary=role_summary,
        band_summary=band_summary,
    ).validate()
