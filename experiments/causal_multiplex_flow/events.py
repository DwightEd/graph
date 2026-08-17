"""Canonical sparse attention to source-aware causal multiplex events.

This module uses only ``ResearchSample`` views.  It intentionally represents
prompt sources by the invariant RP role in Phase 1; no prompt relative-position
centroid or cross-sample semantic alignment is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch


RP = 0
RR = 1


@dataclass(frozen=True)
class EventConfig:
    block_rows: int = 8192
    max_prompt_events_per_token: int = 16
    max_rr_events_per_token: int = 32
    epsilon: float = 1e-8

    def validate(self) -> None:
        if int(self.block_rows) < 1:
            raise ValueError("block_rows must be positive")
        if int(self.max_prompt_events_per_token) < 0:
            raise ValueError("max_prompt_events_per_token must be non-negative")
        if int(self.max_rr_events_per_token) < 1:
            raise ValueError("max_rr_events_per_token must be positive")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class CausalEventSample:
    """Selected retained events and full RP/RR role summaries for one response.

    ``source`` is response-relative for RR events and ``-1`` for RP events.
    Events are sorted by target token; ``target_ptr`` gives the corresponding
    CSR-style token slices.  ``role_summary`` columns are full retained RP mass,
    RR mass, RP edge count, and RR edge count before typed top-k selection.
    """

    sample_id: str
    response_count: int
    num_layers: int
    num_heads: int
    attention_floor: float
    target_ptr: torch.Tensor
    relation: torch.Tensor
    source: torch.Tensor
    channel: torch.Tensor
    weight: torch.Tensor
    lag: torch.Tensor
    role_summary: torch.Tensor

    @property
    def num_channels(self) -> int:
        return int(self.num_layers) * int(self.num_heads)

    @property
    def num_events(self) -> int:
        return int(self.weight.numel())

    def target_slice(self, token_index: int) -> slice:
        token_index = int(token_index)
        if not 0 <= token_index < self.response_count:
            raise IndexError("response token is outside the event sample")
        return slice(
            int(self.target_ptr[token_index].item()),
            int(self.target_ptr[token_index + 1].item()),
        )

    def to(self, device) -> "CausalEventSample":
        return replace(
            self,
            target_ptr=self.target_ptr.to(device),
            relation=self.relation.to(device),
            source=self.source.to(device),
            channel=self.channel.to(device),
            weight=self.weight.to(device),
            lag=self.lag.to(device),
            role_summary=self.role_summary.to(device),
        )

    def validate(self) -> "CausalEventSample":
        if self.response_count < 1:
            raise ValueError("response_count must be positive")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("event geometry must be positive")
        if self.target_ptr.shape != (self.response_count + 1,):
            raise ValueError("target_ptr has the wrong shape")
        if int(self.target_ptr[0]) != 0 or int(self.target_ptr[-1]) != self.num_events:
            raise ValueError("target_ptr does not cover every selected event")
        if bool((self.target_ptr[1:] < self.target_ptr[:-1]).any()):
            raise ValueError("target_ptr must be non-decreasing")
        columns = (self.relation, self.source, self.channel, self.weight, self.lag)
        if any(tensor.ndim != 1 or len(tensor) != self.num_events for tensor in columns):
            raise ValueError("event columns have inconsistent shapes")
        if self.role_summary.shape != (self.response_count, 4):
            raise ValueError("role_summary has the wrong shape")
        if bool(((self.relation != RP) & (self.relation != RR)).any()):
            raise ValueError("unsupported event relation")
        if bool((self.channel < 0).any()) or bool((self.channel >= self.num_channels).any()):
            raise ValueError("event channel is outside the model geometry")
        prompt = self.relation == RP
        response = self.relation == RR
        if bool((self.source[prompt] != -1).any()) or bool((self.lag[prompt] != 0).any()):
            raise ValueError("prompt events must use the role anchor and zero lag")
        if bool((self.source[response] < 0).any()) or bool((self.lag[response] <= 0).any()):
            raise ValueError("RR events must preserve a legal causal source and lag")
        for token in range(self.response_count):
            current = self.target_slice(token)
            if current.start == current.stop:
                continue
            rr_source = self.source[current][self.relation[current] == RR]
            if bool((rr_source >= token).any()):
                raise ValueError("RR event is not strictly causal")
        if not bool(torch.isfinite(self.weight).all()) or bool((self.weight < 0).any()):
            raise ValueError("event weights must be finite and non-negative")
        if not bool(torch.isfinite(self.role_summary).all()) or bool((self.role_summary < 0).any()):
            raise ValueError("role summaries must be finite and non-negative")
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
    """Select strongest events without per-token CPU transfers.

    Canonical CSR order is deterministic. A stable descending weight sort keeps
    that canonical order as the tie break, so repeated extraction is identical
    while remaining on the attention device.
    """
    limit = int(limit)
    if limit <= 0 or indices.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=indices.device)
    order = torch.argsort(
        weight[indices], descending=True, stable=True
    )
    return indices[order[:limit]]


def extract_causal_events(
    sample,
    *,
    config: EventConfig | None = None,
) -> CausalEventSample:
    """Build deterministic typed top-k events from one canonical sample."""
    config = EventConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    channels = layers * heads
    if response_count < 1 or prompt_count < 1:
        raise ValueError("CMRP requires a non-empty prompt and response")
    device = attention.response_values.device

    prompt_mass = torch.zeros(response_count, dtype=torch.float32, device=device)
    rr_mass = torch.zeros_like(prompt_mass)
    prompt_count_full = torch.zeros_like(prompt_mass)
    rr_count_full = torch.zeros_like(prompt_mass)

    target_parts: list[torch.Tensor] = []
    relation_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    channel_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    lag_parts: list[torch.Tensor] = []

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        target = block.query.long()
        channel = (block.layer.long() * heads + block.head.long()).long()
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

        if bool(prompt.any()):
            prompt_mass.index_add_(0, target[prompt], weight[prompt])
            prompt_count_full.index_add_(
                0, target[prompt], torch.ones_like(weight[prompt])
            )
        if bool((~prompt).any()):
            rr_mass.index_add_(0, target[~prompt], weight[~prompt])
            rr_count_full.index_add_(
                0, target[~prompt], torch.ones_like(weight[~prompt])
            )

        target_parts.append(target)
        relation_parts.append(relation)
        source_parts.append(source.long())
        channel_parts.append(channel)
        weight_parts.append(weight)
        lag_parts.append(lag.long())

    if target_parts:
        target_all = torch.cat(target_parts)
        relation_all = torch.cat(relation_parts)
        source_all = torch.cat(source_parts)
        channel_all = torch.cat(channel_parts)
        weight_all = torch.cat(weight_parts)
        lag_all = torch.cat(lag_parts)
        target_order = torch.argsort(target_all, stable=True)
        grouped_count = torch.bincount(target_all, minlength=response_count)
        grouped_ptr = torch.zeros(
            response_count + 1, dtype=torch.long, device=device
        )
        grouped_ptr[1:] = torch.cumsum(grouped_count, dim=0)
    else:
        target_all = torch.empty(0, dtype=torch.long, device=device)
        relation_all = torch.empty_like(target_all)
        source_all = torch.empty_like(target_all)
        channel_all = torch.empty_like(target_all)
        weight_all = torch.empty(0, dtype=torch.float32, device=device)
        lag_all = torch.empty_like(target_all)
        target_order = torch.empty_like(target_all)
        grouped_ptr = torch.zeros(
            response_count + 1, dtype=torch.long, device=device
        )

    selected_parts: list[torch.Tensor] = []
    counts = torch.zeros(response_count, dtype=torch.long, device=device)
    for token in range(response_count):
        current = target_order[
            int(grouped_ptr[token].item()) : int(grouped_ptr[token + 1].item())
        ]
        prompt_indices = current[relation_all[current] == RP]
        rr_indices = current[relation_all[current] == RR]
        selected = torch.cat(
            (
                _stable_top_indices(
                    prompt_indices,
                    weight_all,
                    config.max_prompt_events_per_token,
                ),
                _stable_top_indices(
                    rr_indices,
                    weight_all,
                    config.max_rr_events_per_token,
                ),
            )
        )
        selected_parts.append(selected)
        counts[token] = len(selected)

    selected = (
        torch.cat(selected_parts)
        if selected_parts
        else torch.empty(0, dtype=torch.long, device=device)
    )
    target_ptr = torch.zeros(response_count + 1, dtype=torch.long, device=device)
    target_ptr[1:] = torch.cumsum(counts, dim=0)
    role_summary = torch.stack(
        (prompt_mass, rr_mass, prompt_count_full, rr_count_full), dim=1
    )
    result = CausalEventSample(
        sample_id=str(sample.sample_id),
        response_count=response_count,
        num_layers=layers,
        num_heads=heads,
        attention_floor=float(attention.attention_floor),
        target_ptr=target_ptr,
        relation=relation_all[selected],
        source=source_all[selected],
        channel=channel_all[selected],
        weight=weight_all[selected],
        lag=lag_all[selected],
        role_summary=role_summary,
    )
    return result.validate()
