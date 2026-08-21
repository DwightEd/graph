"""Frozen non-learned baselines used by the source-reuse study."""

from __future__ import annotations

import torch

from .data import SourceReuseGraph


def received_support_topk(graph: SourceReuseGraph, *, topk: int = 5) -> torch.Tensor:
    """Return strict-causal source persistence for every token/layer/head."""

    if topk < 1:
        raise ValueError("topk must be positive")
    channels = graph.num_layers * graph.num_heads
    cumulative = graph.weight.new_zeros((channels, graph.num_response_tokens))
    result = graph.weight.new_zeros((graph.num_response_tokens, channels, topk))

    for token in range(graph.num_response_tokens):
        current = graph.token_slice(token)
        source = graph.source[current]
        selected = source >= graph.response_idx
        if bool(selected.any()):
            response_source = source[selected] - graph.response_idx
            channel = (
                graph.layer[current][selected] * graph.num_heads
                + graph.head[current][selected]
            )
            cumulative.index_put_(
                (channel, response_source),
                graph.weight[current][selected],
                accumulate=True,
            )

        age = token - torch.arange(token + 1, device=graph.device) + 1
        value = cumulative[:, : token + 1] / age.float().unsqueeze(0)
        keep = min(topk, token + 1)
        result[token, :, :keep] = torch.topk(value, keep, dim=1).values

    return result.reshape(
        graph.num_response_tokens,
        graph.num_layers,
        graph.num_heads,
        topk,
    )
