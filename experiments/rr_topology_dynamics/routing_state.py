"""Exact prompt and response routing state for generated tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingState:
    """Attention mass aggregated by exact source-token identity.

    Rows are response queries. Prompt columns use absolute prompt positions;
    response columns use response-relative positions.
    """

    prompt_source_mass: torch.Tensor
    response_source_mass: torch.Tensor
    retained_edge_count: torch.Tensor

    @property
    def response_count(self) -> int:
        return int(self.response_source_mass.shape[0])

    @property
    def prompt_count(self) -> int:
        return int(self.prompt_source_mass.shape[1])


class RoutingStateExtractor:
    """Stream sparse attention blocks into one exact-source routing state."""

    def __init__(self, *, block_rows: int = 8192) -> None:
        if int(block_rows) < 1:
            raise ValueError("block_rows must be positive")
        self.block_rows = int(block_rows)

    def extract(self, sample) -> RoutingState:
        attention = sample.attention()
        response_count = int(attention.num_response_tokens)
        prompt_count = int(attention.response_idx)
        if response_count < 1 or prompt_count < 1:
            raise ValueError("routing state requires non-empty prompt and response")

        attention_values = getattr(attention, "response_values", None)
        device = attention_values.device if attention_values is not None else "cpu"
        prompt_source_mass = torch.zeros(
            (response_count, prompt_count), dtype=torch.float32, device=device
        )
        response_source_mass = torch.zeros(
            (response_count, response_count), dtype=torch.float32, device=device
        )
        retained_edge_count = torch.zeros(
            response_count, dtype=torch.float32, device=device
        )

        for block in sample.iter_sparse_attention_blocks(block_rows=self.block_rows):
            query = block.query.long()
            source = block.source.long()
            weight = block.weight.float()
            if bool((query < 0).any()) or bool((query >= response_count).any()):
                raise ValueError("sparse attention query is outside the response")

            retained_edge_count.index_add_(
                0, query, torch.ones_like(weight, dtype=torch.float32)
            )
            prompt = source < prompt_count
            if bool(prompt.any()):
                prompt_flat_index = query[prompt] * prompt_count + source[prompt]
                prompt_source_mass.view(-1).index_add_(
                    0, prompt_flat_index, weight[prompt]
                )

            response = ~prompt
            if not bool(response.any()):
                continue
            response_source = source[response] - prompt_count
            response_query = query[response]
            if bool((response_source < 0).any()) or bool(
                (response_source >= response_query).any()
            ):
                raise ValueError("sparse attention contains a non-causal response source")
            response_flat_index = response_query * response_count + response_source
            response_source_mass.view(-1).index_add_(
                0, response_flat_index, weight[response]
            )

        return RoutingState(
            prompt_source_mass=prompt_source_mass,
            response_source_mass=response_source_mass,
            retained_edge_count=retained_edge_count,
        )
