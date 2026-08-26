"""Thin adapter around the paper authors' copied GCN and DBGNN classes."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .vendor.dbgnn import HO_GCN
from .vendor.gcn import GCN


UPSTREAM_URL = "https://github.com/lisiq/dbgnn.git"
UPSTREAM_COMMIT = "2613afe5c63183229470164f5decc2bca1a1826e"
ENCODERS = ("dbgnn", "gcn")


class OfficialNodeEncoder(nn.Module):
    """Expose the authors' pre-classifier tensor as a reusable node embedding."""

    def __init__(
        self,
        encoder: str,
        first_order_dim: int,
        higher_order_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}")
        hidden = [hidden_dim, hidden_dim, embedding_dim]
        if encoder == "dbgnn":
            model = HO_GCN(
                num_classes=embedding_dim,
                num_features=[higher_order_dim, first_order_dim],
                hidden_dims=hidden,
                p_dropout=dropout,
            )
        else:
            model = GCN(
                num_features=first_order_dim,
                num_classes=embedding_dim,
                hidden_dims=hidden,
                p_dropout=dropout,
            )
        model.mlp = nn.Identity()
        self.encoder_name = encoder
        self.model = model

    def forward(self, graph) -> torch.Tensor:
        if self.encoder_name == "dbgnn":
            return self.model(graph, graph.x_fo.device)
        first_order = SimpleNamespace(
            x=graph.x_fo,
            edge_index=graph.edge_index_fo,
            edge_weight=graph.edge_weight_fo,
        )
        return self.model(first_order)


class LinkPredictionModel(nn.Module):
    """Train the official encoder without hallucination labels."""

    def __init__(self, encoder: OfficialNodeEncoder, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.source = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.target = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.scale = embedding_dim**-0.5

    def encode(self, graph) -> torch.Tensor:
        return self.encoder(graph)

    def edge_score(
        self,
        embedding: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        left = self.source(embedding[source])
        right = self.target(embedding[target])
        return (left * right).sum(dim=-1) * self.scale
