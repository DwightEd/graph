"""Node-embedding models for the isolated effectiveness audit.

These models deliberately know nothing about the saved edges.  A labelled
``NodeMLP`` measures how readable the frozen representation is, while the
autoencoder and Deep-SVDD heads provide optional label-free alternatives to
the current PCA-kNN detector.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class NodeMLP(nn.Module):
    """A compact supervised probe over frozen node embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(node_embedding).squeeze(-1)


class EmbeddingAutoencoder(nn.Module):
    """Reconstruct frozen node embeddings and score their residual energy."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_embedding)

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(node_embedding))

    def loss(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(node_embedding), node_embedding)

    @torch.no_grad()
    def score(self, node_embedding: torch.Tensor) -> torch.Tensor:
        reconstruction = self(node_embedding)
        return (reconstruction - node_embedding).square().mean(dim=-1)


class DeepSVDD(nn.Module):
    """Map node embeddings into a compact one-class hypersphere."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.empty(0))

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return self.encoder(node_embedding)

    @torch.no_grad()
    def initialize_center(
        self,
        embedding_batches: Iterable[torch.Tensor],
        minimum_magnitude: float = 0.1,
    ) -> torch.Tensor:
        total = torch.zeros(
            self.encoder[-1].out_features,
            device=next(self.parameters()).device,
        )
        count = 0
        for node_embedding in embedding_batches:
            latent = self(node_embedding.to(total.device))
            total += latent.sum(dim=0)
            count += len(latent)
        center = total / count
        signed_floor = torch.where(center < 0, -minimum_magnitude, minimum_magnitude)
        center = torch.where(center.abs() < minimum_magnitude, signed_floor, center)
        self.center = center
        return center

    def loss(self, node_embedding: torch.Tensor) -> torch.Tensor:
        return self.score(node_embedding).mean()

    def score(self, node_embedding: torch.Tensor) -> torch.Tensor:
        latent = self(node_embedding)
        return (latent - self.center).square().sum(dim=-1)
