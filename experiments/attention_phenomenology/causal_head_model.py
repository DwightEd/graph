"""Small head-preserving layer encoder with a causal token-time model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TokenLogits:
    current_logits: torch.Tensor
    next_logits: torch.Tensor


class CausalLayerTemporalModel(nn.Module):
    """Encode ordered layers per token, then model tokens left-to-right."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        num_features: int,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(num_layers, num_heads, num_features, hidden_dim) < 1:
            raise ValueError("model geometry must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_features = int(num_features)

        # Flattening here preserves the fixed identity of every head. A head
        # permutation changes the input coordinates; no head mean is taken.
        self.head_projection = nn.Linear(num_heads * num_features, hidden_dim)
        self.layer_position = nn.Parameter(torch.zeros(num_layers, hidden_dim))
        self.layer_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.current_classifier = nn.Linear(hidden_dim, 1)
        self.next_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> TokenLogits:
        """Score ``[batch, token, layer, head, feature]`` causal sequences."""

        if values.ndim != 5:
            raise ValueError("model input must be [batch, token, layer, head, feature]")
        batch, tokens, layers, heads, features = values.shape
        if (layers, heads, features) != (
            self.num_layers,
            self.num_heads,
            self.num_features,
        ):
            raise ValueError("model input geometry differs from construction")

        per_layer = values.reshape(batch * tokens, layers, heads * features)
        per_layer = torch.tanh(self.head_projection(per_layer))
        per_layer = per_layer + self.layer_position.unsqueeze(0)
        _, final_layer_state = self.layer_encoder(self.dropout(per_layer))
        token_state = final_layer_state[-1].reshape(batch, tokens, -1)
        temporal_state, _ = self.temporal_encoder(self.dropout(token_state))
        temporal_state = self.dropout(temporal_state)
        return TokenLogits(
            current_logits=self.current_classifier(temporal_state).squeeze(-1),
            next_logits=self.next_classifier(temporal_state).squeeze(-1),
        )
