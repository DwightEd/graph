"""Original MAB/ISAB/PMA-style modules for padded source sets."""

from __future__ import annotations

import math

import torch
from torch import nn


class SetMAB(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=float(dropout), batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_padding_mask = None if key_mask is None else ~key_mask.bool()
        attended, _ = self.attention(
            query,
            key_value,
            key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.norm1(query + self.dropout(attended))
        output = self.norm2(output + self.dropout(self.ffn(output)))
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output


class InducedSetAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, induced_points: int, dropout: float) -> None:
        super().__init__()
        self.induced = nn.Parameter(torch.empty(1, int(induced_points), int(dim)))
        nn.init.normal_(self.induced, std=1.0 / math.sqrt(dim))
        self.to_induced = SetMAB(dim, heads, dropout)
        self.to_members = SetMAB(dim, heads, dropout)

    def forward(self, members: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        induced = self.induced.expand(members.shape[0], -1, -1)
        latent = self.to_induced(induced, members, key_mask=mask)
        return self.to_members(members, latent, query_mask=mask)


class PoolingByMultiheadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, seeds: int, dropout: float) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, int(seeds), int(dim)))
        nn.init.normal_(self.seed, std=1.0 / math.sqrt(dim))
        self.pool = SetMAB(dim, heads, dropout)

    def forward(self, members: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seed = self.seed.expand(members.shape[0], -1, -1)
        return self.pool(seed, members, key_mask=mask)


class SetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        induced_points: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    dim, heads, induced_points, dropout
                )
                for _ in range(int(blocks))
            ]
        )
        self.pool = PoolingByMultiheadAttention(
            dim, heads, seeds=1, dropout=dropout
        )

    def forward(
        self, members: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if members.ndim != 3 or mask.shape != members.shape[:2]:
            raise ValueError("set member/mask geometry is inconsistent")
        output = members
        for block in self.blocks:
            output = block(output, mask)
        return output, self.pool(output, mask).squeeze(1)
