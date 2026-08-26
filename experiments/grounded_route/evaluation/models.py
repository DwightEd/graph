"""Small node-only models used by the evaluation module."""

import torch
from torch import nn
from torch.nn import functional as F


class NodeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding).squeeze(-1)


class EmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32) -> None:
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

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(embedding))

    def loss(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(embedding), embedding)

    @torch.no_grad()
    def score(self, embedding: torch.Tensor) -> torch.Tensor:
        return (self(embedding) - embedding).square().mean(dim=-1)


class DeepSVDD(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.encoder(embedding)

    @torch.no_grad()
    def set_center(self, embedding: torch.Tensor) -> None:
        self.center.copy_(self(embedding).mean(dim=0))

    def loss(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.score(embedding).mean()

    def score(self, embedding: torch.Tensor) -> torch.Tensor:
        return (self(embedding) - self.center).square().sum(dim=-1)
