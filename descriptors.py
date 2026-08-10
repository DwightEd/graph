"""Pure graph descriptors for sample-level analysis."""

from collections.abc import Mapping

import torch


def _field(graph, name):
    return graph[name] if isinstance(graph, Mapping) else getattr(graph, name)


def _edge_weights(graph, num_channels: int, device: torch.device) -> torch.Tensor:
    edge_weight = _field(graph, "edge_weight") if (
        (isinstance(graph, Mapping) and "edge_weight" in graph) or hasattr(graph, "edge_weight")
    ) else None
    if edge_weight is not None:
        return torch.as_tensor(edge_weight, device=device, dtype=torch.float32)

    edge_ptr = torch.as_tensor(_field(graph, "edge_ptr"), device=device)
    edge_value = torch.as_tensor(_field(graph, "edge_value"), device=device, dtype=torch.float32)
    counts = edge_ptr[1:] - edge_ptr[:-1]
    edge_ids = torch.repeat_interleave(torch.arange(len(counts), device=device), counts)
    weights = torch.zeros(len(counts), dtype=torch.float32, device=device)
    weights.index_add_(0, edge_ids, edge_value)
    return weights / num_channels


def token_routing_features(graph, num_channels: int) -> torch.Tensor:
    """Return [response_tokens, 4] incoming-routing features for one graph.

    Features are incoming mass, prompt mass share, entropy normalized by the
    number of positive incoming edges, and the history-weighted normalized lag.
    Sparse channel graphs average each edge over every channel, including zeros.
    """
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")

    edge_index = torch.as_tensor(_field(graph, "edge_index"))
    response_idx = int(torch.as_tensor(_field(graph, "response_idx")).item())
    num_nodes = int(torch.as_tensor(_field(graph, "num_nodes")).item())
    response_count = num_nodes - response_idx
    device = edge_index.device
    features = torch.zeros((response_count, 4), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return features

    source, target = edge_index.long()
    if bool(((source < 0) | (source >= num_nodes) | (target < 0) | (target >= num_nodes)).any()):
        raise ValueError("graph edges must reference valid node indices")
    if bool(((target < response_idx) | (source >= target)).any()):
        raise ValueError("graph edges must target response tokens from earlier tokens")
    weights = _edge_weights(graph, num_channels, device)
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("graph edge weights must be finite and non-negative")
    rows = target - response_idx

    total = torch.zeros(response_count, dtype=torch.float32, device=device)
    total.index_add_(0, rows, weights)
    features[:, 0] = total

    prompt_mass = torch.zeros_like(total)
    prompt = source < response_idx
    prompt_mass.index_add_(0, rows[prompt], weights[prompt])
    nonempty = total > 0
    features[nonempty, 1] = prompt_mass[nonempty] / total[nonempty]

    positive = weights > 0
    positive_rows = rows[positive]
    probabilities = weights[positive] / total[positive_rows]
    entropy = torch.zeros_like(total)
    entropy.index_add_(0, positive_rows, -probabilities * probabilities.log())
    counts = torch.bincount(positive_rows, minlength=response_count)
    multiple = counts > 1
    features[multiple, 2] = entropy[multiple] / counts[multiple].float().log()

    history = source >= response_idx
    history_mass = torch.zeros_like(total)
    history_mass.index_add_(0, rows[history], weights[history])
    lags = (target[history] - source[history]).float()
    lag_mass = torch.zeros_like(total)
    lag_mass.index_add_(0, rows[history], weights[history] * lags / max(response_count - 1, 1))
    has_history = history_mass > 0
    features[has_history, 3] = lag_mass[has_history] / history_mass[has_history]
    return features


def temporal_summary(features: torch.Tensor) -> torch.Tensor:
    """Summarize [response_tokens, features] with mean, population std, slope."""
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [response_tokens, features]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    response_count, feature_count = values.shape
    if response_count == 0:
        return torch.zeros(3 * feature_count, dtype=values.dtype, device=values.device)

    mean = values.mean(dim=0)
    std = values.std(dim=0, correction=0)
    if response_count == 1:
        slope = torch.zeros_like(mean)
    else:
        position = torch.linspace(0, 1, response_count, dtype=values.dtype, device=values.device)
        centered_position = position - position.mean()
        slope = (centered_position[:, None] * (values - mean)).sum(dim=0)
        slope /= (centered_position * centered_position).sum()
    return torch.cat((mean, std, slope))
