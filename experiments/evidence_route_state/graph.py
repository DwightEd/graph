"""Complete head-resolved graph state used by the route detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphSequence:
    """Per-token route graph after streaming the complete attention rows.

    The four route channels stay separate throughout this object. Head and
    layer axes likewise remain explicit; the detector, rather than capture,
    decides how differences along those axes contribute to graph distance.
    """

    query_position: torch.Tensor
    prediction_position: torch.Tensor
    node_embedding: torch.Tensor
    residual_gram: torch.Tensor
    head_write_gram: torch.Tensor
    route_topology: torch.Tensor
    mlp_relation: torch.Tensor
    margin_contribution: torch.Tensor
    valid: torch.Tensor


def gram(vectors: torch.Tensor) -> torch.Tensor:
    """Return signed pairwise inner products between route-channel vectors."""

    values = vectors.float()
    return values @ values.transpose(-1, -2)


def route_topology(
    attention: torch.Tensor,
    register_values: torch.Tensor,
    output_gram: torch.Tensor,
    query_position: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Describe every head/channel route using all causal source endpoints.

    ``attention`` is ``[head, query, source]``. ``register_values`` is
    ``[source, channel, kv_head, head_dim]`` and contains the value-space
    contribution of each additive residual register. ``output_gram`` is the
    exact per-query-head ``W_O W_O^T`` metric in head space.

    The seven returned coordinates are log total capacity, log effective
    source count, top-one share, prompt/history/self fractions, and each
    head's Bhattacharyya agreement with the active-head source mixture.
    """

    heads, _queries, sources = attention.shape
    channels = register_values.shape[1]
    kv_heads = register_values.shape[2]
    head_to_kv = torch.arange(heads, device=attention.device) // (heads // kv_heads)

    values = register_values.float()[:, :, head_to_kv].permute(2, 0, 1, 3)
    metric = output_gram.float()
    squared_norm = torch.einsum(
        "hscd,hde,hsce->hsc",
        values,
        metric,
        values,
    )
    source_norm = squared_norm.clamp_min(0).sqrt().permute(0, 2, 1)
    capacity = attention.float().permute(1, 0, 2)[:, :, None]
    capacity = capacity * source_norm[None]

    source = torch.arange(sources, device=attention.device)
    query = query_position.to(device=attention.device, dtype=torch.long)
    causal = source[None] <= query[:, None]
    capacity = capacity * causal[:, None, None]

    total = capacity.sum(-1)
    row_scale = total.sum(-1, keepdim=True)
    active = total > (1e-6 * row_scale).clamp_min(1e-12)
    probability = capacity / total.clamp_min(1e-12)[..., None]
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    effective_sources = entropy.exp().masked_fill(~active, 0)
    top_one = probability.amax(-1).masked_fill(~active, 0)

    self_endpoint = source[None] == query[:, None]
    prompt_endpoint = (source[None] < response_start) & ~self_endpoint
    history_endpoint = (source[None] >= response_start) & (
        source[None] < query[:, None]
    )

    def fraction(endpoint: torch.Tensor) -> torch.Tensor:
        selected = (capacity * endpoint[:, None, None]).sum(-1)
        return (selected / total.clamp_min(1e-12)).masked_fill(~active, 0)

    active_count = active.sum(1).clamp_min(1)
    mixture = (probability * active[..., None]).sum(1)
    mixture = mixture / active_count[:, :, None]
    consensus = torch.sqrt(probability * mixture[:, None].clamp_min(0)).sum(-1)
    consensus = consensus.masked_fill(~active, 0)

    return torch.stack(
        (
            total.log1p(),
            effective_sources.log1p(),
            top_one,
            fraction(prompt_endpoint),
            fraction(history_endpoint),
            fraction(self_endpoint),
            consensus,
        ),
        dim=-1,
    ).reshape(len(query), heads, channels, 7)


def mlp_relation(registers: torch.Tensor, mlp_write: torch.Tensor) -> torch.Tensor:
    """Relate the native MLP write to each pre-MLP route register.

    Signed cosines expose whether the same-token nonlinear update reinforces
    or opposes each route direction. The last coordinate is its log relative
    norm against the complete pre-MLP residual state.
    """

    state = registers.float()
    write = mlp_write.float()
    state_norm = state.norm(dim=-1)
    write_norm = write.norm(dim=-1)
    cosine = (state * write[:, None]).sum(-1)
    cosine = cosine / (state_norm * write_norm[:, None]).clamp_min(1e-12)
    complete_state_norm = state.sum(1).norm(dim=-1)
    negligible = state_norm <= 1e-6 * complete_state_norm[:, None]
    cosine = cosine.masked_fill(negligible | (write_norm[:, None] == 0), 0)

    relative_norm = (write_norm / complete_state_norm.clamp_min(1e-12)).log1p()
    return torch.cat((cosine, relative_norm[:, None]), dim=1)
