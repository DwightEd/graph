"""A small source-graph residual adapter for frozen pre-token states.

Attention is an owner prediction, never an observed native route. The adapter
does not contain an LM, tokenizer, semantic verifier, or hallucination labels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class GroundedGraphAdapter(nn.Module):
    """All learned graph/pointer/payload transformations form one adapter.

    source_states: [N,D], pooled only from the original prompt's source tokens.
    query_states: [T,D], original replay state BEFORE each scored target token.
    edge_index: [2,E], source/destination node indices, no absolute-ID feature.
    graph_ids: [N] and query_graph_ids: [T] prohibit cross-sample attention.
    source_available: [N] marks only mapped source nodes, not semantic truth.
    """

    def __init__(self, input_dim=4096, adapter_dim=128, node_types=9, edge_types=4, message_steps=2):
        super().__init__()
        if min(input_dim, adapter_dim, node_types, edge_types, message_steps) < 1:
            raise ValueError("all adapter dimensions must be positive")
        self.input_dim, self.adapter_dim = input_dim, adapter_dim
        self.node_projection = nn.Linear(input_dim, adapter_dim, bias=False)
        self.query_projection = nn.Linear(input_dim, adapter_dim, bias=False)
        self.node_type = nn.Embedding(node_types, adapter_dim)
        self.edge_type = nn.Embedding(edge_types, adapter_dim)
        self.input_norm = nn.LayerNorm(adapter_dim)
        self.message = nn.ModuleList(nn.Linear(adapter_dim, adapter_dim, bias=False) for _ in range(message_steps))
        self.message_norm = nn.ModuleList(nn.LayerNorm(adapter_dim) for _ in range(message_steps))
        self.key = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.value = nn.Linear(adapter_dim, adapter_dim, bias=False)
        self.output = nn.Linear(adapter_dim, input_dim, bias=False)
        self.gate = nn.Linear(2 * adapter_dim, 1)
        # Start at the frozen model; pointer loss still trains graph/query paths.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, source_states, query_states, node_types, edge_index, edge_types,
                graph_ids, query_graph_ids, source_available=None, *, use_edges=True):
        if (source_states.ndim != 2 or query_states.ndim != 2
                or source_states.shape[1] != self.input_dim or query_states.shape[1] != self.input_dim):
            raise ValueError("source/query dimensions differ from frozen backbone")
        n, t = source_states.shape[0], query_states.shape[0]
        if n < 1 or t < 1:
            raise ValueError("nonempty source and query states required")
        if not source_states.is_floating_point() or not query_states.is_floating_point():
            raise TypeError("frozen source/query features must be floating point")
        weight = self.node_projection.weight
        tensors = (source_states, query_states, node_types, edge_index, edge_types, graph_ids, query_graph_ids)
        if any(x.device != weight.device for x in tensors):
            raise ValueError("move graph/features and adapter to the same device explicitly")
        # Capture stores bf16; trainable small adapters normally use float32.
        source_states = source_states.to(weight.dtype)
        query_states = query_states.to(weight.dtype)
        if (node_types.shape != (n,) or graph_ids.shape != (n,) or query_graph_ids.shape != (t,)
                or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_types.shape != (edge_index.shape[1],)):
            raise ValueError("node/edge/graph membership dimensions differ")
        if any(x.dtype != torch.long for x in (node_types, edge_index, edge_types, graph_ids, query_graph_ids)):
            raise TypeError("topology and graph membership tensors must be int64")
        if (torch.any(node_types < 0) or torch.any(node_types >= self.node_type.num_embeddings)
                or torch.any(edge_types < 0) or torch.any(edge_types >= self.edge_type.num_embeddings)
                or torch.any(edge_index < 0) or torch.any(edge_index >= n)):
            raise ValueError("node/edge type or endpoint is out of bounds")
        src, dst = edge_index
        if torch.any(graph_ids[src] != graph_ids[dst]):
            raise ValueError("source graph edge crosses independent samples")
        if source_available is None:
            source_available = torch.ones(n, dtype=torch.bool, device=source_states.device)
        if source_available.shape != (n,) or source_available.dtype != torch.bool:
            raise ValueError("source availability must be a node boolean mask")
        if source_available.device != weight.device:
            raise ValueError("availability mask is on a different device")
        eligible = (query_graph_ids[:, None] == graph_ids[None, :]) & source_available[None, :]
        if not torch.all(eligible.any(dim=1)):
            raise ValueError("query has no available source node; caller must emit unavailable")
        if not torch.isfinite(source_states).all() or not torch.isfinite(query_states).all():
            raise ValueError("nonfinite frozen features")
        z = self.input_norm(self.node_projection(source_states) + self.node_type(node_types))
        if use_edges and src.numel():
            usable = source_available[src] & source_available[dst]
            src, dst, types = src[usable], dst[usable], edge_types[usable]
            degree = torch.zeros(n, device=z.device, dtype=z.dtype).index_add_(
                0, dst, torch.ones_like(dst, dtype=z.dtype)).clamp_min(1)[:, None]
            for transform, norm in zip(self.message, self.message_norm, strict=True):
                messages = F.silu(transform(z[src]) + self.edge_type(types))
                summed = torch.zeros_like(z).index_add_(0, dst, messages)
                z = norm(z + summed / degree)
        q = self.query_projection(query_states)
        logits = (q @ self.key(z).T) / math.sqrt(self.adapter_dim)
        logits = logits.masked_fill(~eligible, -torch.inf)
        attention = logits.softmax(-1)
        context = attention @ self.value(z)
        gate = torch.sigmoid(self.gate(torch.cat([q, context], dim=-1)))
        residual = gate * self.output(context)
        return {"hidden": query_states + residual, "pointer_logits": logits,
            "pointer": attention, "gate": gate.squeeze(-1), "residual": residual,
            "node_states": z, "eligible": eligible}


def pointer_loss(output, positives, target_queries):
    """Multi-positive pointer NLL; supervision provenance belongs to caller.

    A joint evidence bundle is ONE inventory node. Multiple positives here mean
    interchangeable weak labels, not an AND over necessary evidence members.
    """
    logits, eligible = output["pointer_logits"], output["eligible"]
    if (positives.shape != logits.shape or positives.dtype != torch.bool
            or target_queries.shape != (logits.shape[0],) or target_queries.dtype != torch.bool):
        raise ValueError("pointer supervision masks differ")
    if torch.any(positives & ~eligible):
        raise ValueError("positive source owner escapes its sample or is unavailable")
    if torch.any(target_queries & ~positives.any(dim=1)):
        raise ValueError("supervised query has no positive; do not invent an owner")
    if not target_queries.any():
        return output["hidden"].sum() * 0.
    selected = logits[target_queries]
    positive = selected.masked_fill(~positives[target_queries], -torch.inf)
    return (selected.logsumexp(-1) - positive.logsumexp(-1)).mean()


def token_loss(hidden, target_ids, frozen_lm_weight, *, chunk_size=64):
    """Exact full-vocabulary CE through frozen unembedding; no sampled labels."""
    if frozen_lm_weight.requires_grad:
        raise ValueError("LM head must remain frozen")
    if (hidden.ndim != 2 or frozen_lm_weight.ndim != 2
            or hidden.shape[1] != frozen_lm_weight.shape[1] or target_ids.shape != (hidden.shape[0],)
            or target_ids.dtype != torch.long or hidden.shape[0] == 0 or chunk_size < 1):
        raise ValueError("invalid grounded token loss dimensions")
    if torch.any(target_ids < 0) or torch.any(target_ids >= frozen_lm_weight.shape[0]):
        raise ValueError("target outside frozen vocabulary")
    def chunk_loss(states, labels):
        logits = F.linear(states.to(frozen_lm_weight.dtype), frozen_lm_weight).float()
        return F.cross_entropy(logits, labels, reduction="sum")

    losses = []
    for start in range(0, len(hidden), chunk_size):
        stop = min(start + chunk_size, len(hidden))
        # Checkpoint CE as well as projection: merely splitting then summing
        # losses retains all T×vocabulary activations until backward.
        if torch.is_grad_enabled() and hidden.requires_grad:
            loss = checkpoint(chunk_loss, hidden[start:stop], target_ids[start:stop], use_reentrant=False)
        else:
            loss = chunk_loss(hidden[start:stop], target_ids[start:stop])
        losses.append(loss)
    return torch.stack(losses).sum() / len(hidden)
