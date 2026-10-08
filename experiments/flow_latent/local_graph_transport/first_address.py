"""First-choice source address versus summed-payload statistical readers.

All variants retain every archived node/head coordinate and the same parameters.
The group control uses native summed AV with pooled-key conditioning; address
variants instead select individual contextual source K/V. This is a learned
reader of fixed captures, not a native attention intervention.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def reverse_valid_payload(value, valid):
    """Keep each key and padding slot fixed; reverse only its valid payloads."""
    lengths = valid.sum(-1, keepdim=True)
    positions = torch.arange(value.shape[1], device=value.device)[None]
    reversed_positions = (lengths - 1 - positions).clamp_min(0)
    reversed_positions = torch.where(valid, reversed_positions, positions)
    return value.gather(1, reversed_positions[..., None].expand_as(value))


class FirstAddressReader(nn.Module):
    """Two candidates share a single receiver embedding and dropout realization."""

    def __init__(self, model_dim=4096, source_dim=1024, address_dim=64,
                 payload_dim=128, dropout=.1, temperature=16.,
                 address_energy='dot', address_temperature=8.):
        super().__init__()
        self.node_projection = nn.Linear(6 * model_dim, 64)
        self.boundary_projection = nn.Linear(10 * model_dim, 64)
        self.receiver_semantic = nn.Linear(128, model_dim)
        self.direct_weights = nn.Parameter(torch.zeros(6, model_dim))
        self.query_norm = nn.LayerNorm(model_dim)
        self.key_norm = nn.LayerNorm(source_dim)
        self.value_norm = nn.LayerNorm(source_dim)
        self.address_query = nn.Linear(model_dim, address_dim, bias=False)
        self.address_key = nn.Linear(source_dim, address_dim, bias=False)
        self.payload_value = nn.Linear(source_dim, payload_dim, bias=False)
        self.source_semantic = nn.Linear(payload_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(address_dim)
        self.temperature = temperature
        self.address_energy = address_energy
        self.address_temperature = address_temperature

    def source_context(self, query, key, value, valid, summed_payload, variant):
        """Same query/key/payload maps are operative in every control."""
        query = self.address_query(self.query_norm(query))
        keys = self.address_key(self.key_norm(key))
        if variant == 'group_summed':
            pooled_key = (keys * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            if self.address_energy == 'cosine':
                energy = self.address_temperature * (F.normalize(query, dim=-1) *
                    F.normalize(pooled_key, dim=-1)).sum(-1)
            else:
                energy = (query * pooled_key).sum(-1) / self.scale
            gate = energy.sigmoid()
            payload = self.payload_value(self.value_norm(summed_payload))
            return gate[:, None] * payload, None

        if variant == 'payload_rewired':
            value = reverse_valid_payload(value, valid)
        elif variant != 'address':
            raise ValueError(variant)
        payload = self.payload_value(self.value_norm(value))
        if self.address_energy == 'cosine':
            scores = self.address_temperature * torch.einsum('bd,bsd->bs',
                F.normalize(query, dim=-1), F.normalize(keys, dim=-1))
        else:
            scores = torch.einsum('bd,bsd->bs', query, keys) / self.scale
        weights = scores.masked_fill(~valid, -torch.inf).softmax(-1)
        return torch.einsum('bs,bsp->bp', weights, payload), weights

    def forward(self, node_fields, boundary_fields, query, source_key,
                source_value, source_valid, summed_payload, candidate_vectors,
                variant='address', return_attention=False):
        """Risk [graph,2]: candidate vectors never enter the address weights."""
        node = F.gelu(self.node_projection(node_fields.flatten(1)))
        boundary = F.gelu(self.boundary_projection(boundary_fields.flatten(1)))
        receiver = self.dropout(torch.cat([node, boundary], dim=-1))
        direct = torch.einsum('bfd,fd->bd', node_fields, self.direct_weights)
        context, weights = self.source_context(query, source_key, source_value,
                                               source_valid, summed_payload, variant)
        semantic = self.receiver_semantic(receiver) + direct + self.source_semantic(context)
        semantic = F.normalize(semantic, dim=-1)
        candidates = F.normalize(candidate_vectors, dim=-1)
        risk = -self.temperature * torch.einsum('bd,bcd->bc', semantic, candidates)
        return (risk, weights) if return_attention else risk


def first_choice_loss(risk, pair_weight=1.):
    """Column 0 is compatible; column 1 is incompatible with identical history."""
    labels = risk.new_tensor([0., 1.]).expand_as(risk)
    bce = F.binary_cross_entropy_with_logits(risk, labels)
    pair = F.softplus(risk[:, 0] - risk[:, 1]).mean()
    return bce + pair_weight * pair
