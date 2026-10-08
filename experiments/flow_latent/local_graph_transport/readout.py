"""Supervised full-coordinate node and paired local-message readout.

The trainable head factors carry native, content, and routing coordinates into
a statistical latent representation. They do not implement native W_O transport
or establish a causal information path. Input normalization is fitted externally
on training sources; variants share exactly the same trainable parameters.
"""
import math

import torch
from torch import nn


def rewire_local_weights(attention, valid):
    """Swap valid weights within lags 3/4 and 5..8; keep every sender fixed.

    Attention is [world, row, head, lag]. Both worlds use the same permutation.
    Lags 1/2, invalid entries, and each world's per-head mass are preserved.
    """
    result = attention.clone()
    for lower, upper in ((2, 4), (4, 8)):
        upper = min(upper, attention.shape[-1])
        if lower >= upper:
            continue
        selected = valid[:, lower:upper]
        columns = torch.arange(lower, upper, device=attention.device).expand(len(valid), -1)
        ordered = columns.masked_fill(~selected, upper).sort(-1).values
        rank = selected.long().cumsum(-1) - 1
        reverse_rank = (selected.sum(-1, keepdim=True) - 1 - rank).clamp_min(0)
        mapping = ordered.gather(1, reverse_rank)
        mapping = torch.where(selected, mapping, columns)
        gather = mapping[None, :, None].expand(attention.shape[0], -1, attention.shape[2], -1)
        result[..., lower:upper] = attention.gather(3, gather)
    return result


def graph_inputs(attention, indices, valid, variant):
    """Prepare matched graph controls without modifying endpoint node fields."""
    weights = attention * valid[None, :, None]
    senders = indices
    if variant == 'rewired':
        weights = rewire_local_weights(weights, valid)
    elif variant == 'uniform':
        mass = weights.sum(-1, keepdim=True)
        count = valid.sum(-1).clamp_min(1)[None, :, None, None]
        weights = mass / count * valid[None, :, None]
    elif variant == 'self':
        senders = torch.arange(len(indices), device=indices.device)[:, None].expand_as(indices)
    elif variant != 'real':
        raise ValueError(f'Unknown graph control: {variant}')
    return weights, senders.clamp_min(0)


def edge_factors(value_fields, attention, indices):
    """Return every edge's signed [native AV, meanA deltaV, deltaA meanV].

    Values: [row, head, 2*head_dim], concatenated native V and native-minus-
    blocked V. Attention: [2, row, head, lag], native then blocked. Output:
    [row, lag, head, 3*head_dim]. Sender indices are relative to stored rows.
    """
    values = value_fields[indices]
    native_values, value_difference = values.chunk(2, dim=-1)
    native_attention = attention[0].transpose(1, 2)[..., None]
    blocked_attention = attention[1].transpose(1, 2)[..., None]
    mean_attention = (native_attention + blocked_attention) / 2
    attention_difference = native_attention - blocked_attention
    mean_values = native_values - value_difference / 2
    native = native_attention * native_values
    content = mean_attention * value_difference
    routing = attention_difference * mean_values
    return torch.cat([native, content, routing], dim=-1)


class LocalTransportReader(nn.Module):
    """BCE-ready per-token logits from full nodes, boundaries, and actual edges."""

    def __init__(self, model_dim=4096, heads=32, head_dim=128, scalar_dim=8,
                 dropout=.1, use_gate=True):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.use_gate = use_gate
        self.node_projection = nn.Linear(6 * model_dim, 32)
        self.boundary_projection = nn.Linear(10 * model_dim, 32)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head_projection = nn.Parameter(torch.empty(heads, 3 * head_dim, 8))
        nn.init.normal_(self.head_projection, std=1 / math.sqrt(3 * head_dim))
        self.sender_projection = nn.Linear(64, heads * 8, bias=False)
        self.receiver_gate = nn.Linear(128, heads)
        self.graph_projection = nn.Linear(heads * 8, 64)
        self.classifier = nn.Sequential(nn.Linear(128 + scalar_dim, 64), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(64, 1))

    def node_embedding(self, node_fields, boundary_fields):
        native = self.activation(self.node_projection(node_fields.flatten(1)))
        boundary = self.activation(self.boundary_projection(boundary_fields.flatten(1)))
        return self.dropout(torch.cat([native, boundary], dim=-1))

    def transport_embedding(self, nodes, value_fields, attention, indices, valid, variant):
        weights, senders = graph_inputs(attention, indices, valid, variant)
        factors = edge_factors(value_fields, weights, senders)
        edge = torch.einsum('rwhd,hdc->rwhc', factors, self.head_projection)
        sender_nodes = nodes[senders]
        sender_readout = self.sender_projection(sender_nodes).reshape(
            len(nodes), indices.shape[1], self.heads, 8)
        native_weight = weights[0].transpose(1, 2)[..., None]
        edge = edge + native_weight * sender_readout
        if self.use_gate:
            receivers = nodes[:, None].expand(-1, indices.shape[1], -1)
            gate = self.receiver_gate(torch.cat([receivers, sender_nodes], dim=-1)).sigmoid()
            edge = edge * gate[..., None]
        message = edge.sum(dim=1).flatten(1)
        return self.dropout(self.activation(self.graph_projection(message)))

    def forward(self, node_fields, boundary_fields, value_fields, local_attention,
                indices, valid, scalars, variant='real', return_embedding=False):
        """Each returned logit belongs to its own row; no span score broadcasting."""
        nodes = self.node_embedding(node_fields, boundary_fields)
        graph = self.transport_embedding(nodes, value_fields, local_attention,
                                          indices, valid, variant)
        embedding = torch.cat([nodes, graph, scalars], dim=-1)
        logits = self.classifier(embedding).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits
