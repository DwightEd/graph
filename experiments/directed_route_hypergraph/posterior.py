"""Deterministic and variational bottlenecks for route-node states."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PosteriorOutput:
    decoder_embedding: torch.Tensor
    exported_embedding: torch.Tensor
    mean: torch.Tensor
    log_variance: torch.Tensor


class VariationalRoutePosterior(nn.Module):
    """Map final route states to Gaussian endpoint/layout decoder latents."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        export: str,
        logvar_min: float,
        logvar_max: float,
    ) -> None:
        super().__init__()
        if export not in {"mean", "mean_logvar"}:
            raise ValueError("vae export must be 'mean' or 'mean_logvar'")
        self.export = export
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, hidden_dim)
        self.log_variance = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_variance.weight)
        nn.init.constant_(self.log_variance.bias, -2.0)

    def forward(self, state: torch.Tensor) -> PosteriorOutput:
        normalized = self.input_norm(state)
        mean = state + self.mean(normalized)
        log_variance = self.log_variance(normalized).clamp(
            self.logvar_min,
            self.logvar_max,
        )
        if self.training:
            noise = torch.randn_like(mean)
            decoder_embedding = mean + noise * torch.exp(0.5 * log_variance)
        else:
            decoder_embedding = mean
        exported = (
            mean
            if self.export == "mean"
            else torch.cat((mean, log_variance), dim=-1)
        )
        return PosteriorOutput(
            decoder_embedding=decoder_embedding,
            exported_embedding=exported,
            mean=mean,
            log_variance=log_variance,
        )


def deterministic_posterior(state: torch.Tensor) -> PosteriorOutput:
    zeros = torch.zeros_like(state)
    return PosteriorOutput(
        decoder_embedding=state,
        exported_embedding=state,
        mean=state,
        log_variance=zeros,
    )
