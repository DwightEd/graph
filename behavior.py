"""Token-level behavior analysis for attention graphs.

This module keeps the existing t-SNE descriptors unchanged and adds features
for case studies around labeled hallucination spans. All label-dependent
operations are separated from graph feature extraction.
"""

from collections.abc import Sequence

import torch

from descriptors import _field, token_routing_features


BEHAVIOR_FEATURE_NAMES = (
    "incoming_mass",
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_degree",
    "prompt_degree",
    "history_degree",
    "in_density",
    "prompt_density",
    "history_density",
    "history_edge_share",
)
WINDOW_NAMES = ("pre", "error", "post")


def token_behavior_features(graph, num_channels: int) -> torch.Tensor:
    """Return [response_tokens, 11] original-threshold routing/topology features.

    The first four columns are exactly ``token_routing_features``. The added
    columns describe threshold-retained edge counts and length-normalized
    densities; they are not meaningful for cardinality-capped top-k graphs.
    """
    routing = token_routing_features(graph, num_channels)
    edge_index = torch.as_tensor(_field(graph, "edge_index"))
    response_idx = int(torch.as_tensor(_field(graph, "response_idx")).item())
    num_nodes = int(torch.as_tensor(_field(graph, "num_nodes")).item())
    response_count = num_nodes - response_idx
    device = edge_index.device
    topology = torch.zeros((response_count, 7), dtype=torch.float32, device=device)
    if edge_index.numel() == 0:
        return torch.cat((routing, topology), dim=1)

    source, target = edge_index.long()
    rows = target - response_idx
    prompt = source < response_idx
    history = ~prompt

    in_degree = torch.bincount(rows, minlength=response_count).float()
    prompt_degree = torch.bincount(rows[prompt], minlength=response_count).float()
    history_degree = torch.bincount(rows[history], minlength=response_count).float()

    targets = torch.arange(response_idx, num_nodes, device=device, dtype=torch.float32)
    possible_history = targets - float(response_idx)
    in_density = in_degree / targets.clamp_min(1.0)
    prompt_density = prompt_degree / float(max(response_idx, 1))
    history_density = torch.zeros_like(history_degree)
    has_history = possible_history > 0
    history_density[has_history] = history_degree[has_history] / possible_history[has_history]
    history_edge_share = torch.zeros_like(history_degree)
    nonempty = in_degree > 0
    history_edge_share[nonempty] = history_degree[nonempty] / in_degree[nonempty]

    topology[:, 0] = in_degree
    topology[:, 1] = prompt_degree
    topology[:, 2] = history_degree
    topology[:, 3] = in_density
    topology[:, 4] = prompt_density
    topology[:, 5] = history_density
    topology[:, 6] = history_edge_share
    return torch.cat((routing, topology), dim=1)


def validate_positive_runs(response_count: int, runs: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Validate sorted, non-overlapping ``[start, end)`` response spans."""
    if response_count < 0:
        raise ValueError("response_count must be non-negative")
    normalized: list[tuple[int, int]] = []
    previous_end = 0
    for run in runs:
        if len(run) != 2:
            raise ValueError("each positive run must contain [start, end)")
        start, end = int(run[0]), int(run[1])
        if not 0 <= start < end <= response_count:
            raise ValueError("positive run is outside the response")
        if normalized and start < previous_end:
            raise ValueError("positive runs must be sorted and non-overlapping")
        normalized.append((start, end))
        previous_end = end
    return tuple(normalized)


def positive_mask(response_count: int, runs: Sequence[Sequence[int]], *, device=None) -> torch.Tensor:
    """Return a boolean response-token mask for hallucination spans."""
    normalized = validate_positive_runs(response_count, runs)
    mask = torch.zeros(response_count, dtype=torch.bool, device=device)
    for start, end in normalized:
        mask[start:end] = True
    return mask


def centered_window(features: torch.Tensor, center: int, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a NaN-padded window and validity mask centered on one token."""
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [response_tokens, features]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not 0 <= center < len(values):
        raise ValueError("center must index a response token")

    width = 2 * radius + 1
    output = torch.full((width, values.shape[1]), float("nan"), dtype=values.dtype, device=values.device)
    valid = torch.zeros(width, dtype=torch.bool, device=values.device)
    source_start = max(0, center - radius)
    source_end = min(len(values), center + radius + 1)
    target_start = source_start - (center - radius)
    target_end = target_start + (source_end - source_start)
    output[target_start:target_end] = values[source_start:source_end]
    valid[target_start:target_end] = True
    return output, valid


def align_error_onsets(
    features: torch.Tensor,
    runs: Sequence[Sequence[int]],
    radius: int,
    *,
    policy: str = "first",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align one sample's behavior trajectories around hallucination onset(s)."""
    normalized = validate_positive_runs(len(features), runs)
    if policy not in {"first", "all"}:
        raise ValueError("policy must be 'first' or 'all'")
    selected = normalized[:1] if policy == "first" else normalized
    if not selected:
        width = 2 * radius + 1
        return (
            torch.empty((0, width, features.shape[1]), dtype=torch.float32, device=features.device),
            torch.empty((0, width), dtype=torch.bool, device=features.device),
        )
    windows, valid = zip(*(centered_window(features, start, radius) for start, _ in selected))
    return torch.stack(windows), torch.stack(valid)


def summarize_run_windows(
    features: torch.Tensor,
    runs: Sequence[Sequence[int]],
    *,
    pre_window: int = 8,
    post_window: int = 8,
) -> torch.Tensor:
    """Return [runs, pre/error/post, features] means with NaN for empty windows."""
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [response_tokens, features]")
    if pre_window < 0 or post_window < 0:
        raise ValueError("window sizes must be non-negative")
    normalized = validate_positive_runs(len(values), runs)
    output = torch.full((len(normalized), 3, values.shape[1]), float("nan"), dtype=values.dtype, device=values.device)

    for row, (start, end) in enumerate(normalized):
        segments = (
            values[max(0, start - pre_window):start],
            values[start:end],
            values[end:min(len(values), end + post_window)],
        )
        for column, segment in enumerate(segments):
            if len(segment):
                output[row, column] = segment.mean(dim=0)
    return output
