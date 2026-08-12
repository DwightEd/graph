"""Small utilities for response-label spans and onset-aligned windows."""

from collections.abc import Sequence

import torch


WINDOW_NAMES = ("pre", "error", "post")


def validate_positive_runs(
    response_count: int, runs: Sequence[Sequence[int]]
) -> tuple[tuple[int, int], ...]:
    """Validate sorted, non-overlapping [start, end) response spans."""
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


def positive_mask(response_count: int, runs, *, device=None) -> torch.Tensor:
    mask = torch.zeros(response_count, dtype=torch.bool, device=device)
    for start, end in validate_positive_runs(response_count, runs):
        mask[start:end] = True
    return mask


def centered_window(
    features: torch.Tensor, center: int, radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a NaN-padded feature window and its validity mask."""
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
    output = torch.full(
        (width, values.shape[1]), float("nan"), dtype=values.dtype, device=values.device
    )
    valid = torch.zeros(width, dtype=torch.bool, device=values.device)
    source_start = max(0, center - radius)
    source_end = min(len(values), center + radius + 1)
    target_start = source_start - (center - radius)
    target_end = target_start + (source_end - source_start)
    output[target_start:target_end] = values[source_start:source_end]
    valid[target_start:target_end] = True
    return output, valid


def align_error_onsets(features, runs, radius: int, *, policy="first"):
    """Align feature trajectories around one or all labeled error onsets."""
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
    windows, valid = zip(
        *(centered_window(features, start, radius) for start, _ in selected)
    )
    return torch.stack(windows), torch.stack(valid)


def summarize_run_windows(
    features: torch.Tensor,
    runs,
    *,
    pre_window: int = 8,
    post_window: int = 8,
) -> torch.Tensor:
    """Return [runs, pre/error/post, features] means."""
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError("features must have shape [response_tokens, features]")
    if pre_window < 0 or post_window < 0:
        raise ValueError("window sizes must be non-negative")
    normalized = validate_positive_runs(len(values), runs)
    output = torch.full(
        (len(normalized), 3, values.shape[1]),
        float("nan"),
        dtype=values.dtype,
        device=values.device,
    )
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
