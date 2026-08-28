"""Numerically bind replay attention to the frozen sparse attention cache.

The functional audit multiplies cached attention weights by value states and
gradients from a fresh teacher-forced replay.  Matching only the architecture
or checkpoint name is not enough: a different checkpoint would create a
scientifically meaningless hybrid quantity.  This module therefore compares
every retained cache endpoint and every exact response diagonal against the
replay before any functional feature is accepted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


DEFAULT_ATTENTION_ATOL = 5e-3


def _numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _selected_attention(
    attention: Any,
    head: np.ndarray,
    query: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    """Select sparse endpoints without copying a dense GPU matrix to CPU."""

    # NumPy arrays also expose a ``device`` attribute in recent releases;
    # tensor semantics require detach/to-device, not that attribute alone.
    if hasattr(attention, "detach") and hasattr(attention, "device"):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - runtime mismatch
            raise RuntimeError("Torch attention tensor is unavailable") from error
        device = attention.device
        selected = attention[
            torch.as_tensor(head, dtype=torch.long, device=device),
            torch.as_tensor(query, dtype=torch.long, device=device),
            torch.as_tensor(source, dtype=torch.long, device=device),
        ]
        return selected.detach().float().cpu().numpy()
    return np.asarray(attention)[head, query, source].astype(np.float64, copy=False)


def _channel(value: Any, *, heads: int, tokens: int) -> Any:
    shape = tuple(value.shape)
    if shape == (1, heads, tokens, tokens):
        return value[0]
    if shape == (heads, tokens, tokens):
        return value
    raise ValueError(
        "replay attention must have shape [1, head, token, token] or "
        "[head, token, token]"
    )


@dataclass(frozen=True)
class AttentionCacheBinding:
    """A compact audit record for one cache/replay numerical comparison."""

    verified: bool
    absolute_tolerance: float
    retained_endpoints_compared: int
    diagonal_endpoints_compared: int
    retained_max_abs_error: float
    diagonal_max_abs_error: float
    known_mass_max_abs_error: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_replay_attention(
    graph: Any,
    attentions: Sequence[Any],
    *,
    absolute_tolerance: float = DEFAULT_ATTENTION_ATOL,
) -> AttentionCacheBinding:
    """Require replay and cache attention to agree on every observed endpoint.

    The formal cache censors small off-diagonal values, so it is impossible to
    compare every dense entry.  Retained off-diagonal endpoints and the exact
    diagonal are sufficient to reject a different checkpoint, mask, token
    alignment, head order, or layer order.  The retained-plus-diagonal row mass
    is checked independently against ``1 - unresolved``.
    """

    tolerance = float(absolute_tolerance)
    if not 0.0 < tolerance < 0.1:
        raise ValueError("attention binding tolerance must lie in (0, 0.1)")
    if len(attentions) != int(graph.layer_count):
        raise ValueError("replay attention layer count differs from the cache")

    response_count = int(graph.response_count)
    layer_count = int(graph.layer_count)
    head_count = int(graph.head_count)
    token_count = int(graph.token_count)
    response_start = int(graph.response_start)
    cached_diagonal = _numpy(graph.diagonal, dtype=np.float64)
    cached_unresolved = _numpy(graph.unresolved, dtype=np.float64)
    if cached_diagonal.shape != (response_count, layer_count, head_count):
        raise ValueError("cached diagonal has invalid geometry")
    if cached_unresolved.shape != cached_diagonal.shape:
        raise ValueError("cached unresolved mass has invalid geometry")

    retained_errors: list[np.ndarray] = []
    diagonal_errors: list[np.ndarray] = []
    known_mass_errors: list[np.ndarray] = []
    retained_count = 0
    diagonal_count = response_count * layer_count * head_count

    for layer in range(layer_count):
        channel = _channel(
            attentions[layer], heads=head_count, tokens=token_count
        )
        edges = graph.layer_edges(layer)
        source = _numpy(edges.source, dtype=np.int64)
        target = _numpy(edges.target, dtype=np.int64)
        head = _numpy(edges.head, dtype=np.int64)
        cached_weight = _numpy(edges.weight, dtype=np.float64)
        if not (
            source.shape == target.shape == head.shape == cached_weight.shape
        ):
            raise ValueError("cached sparse endpoints are misaligned")
        if source.size:
            replay_weight = _selected_attention(channel, head, target, source)
            if not np.isfinite(replay_weight).all():
                raise ValueError("replay retained attention contains non-finite values")
            retained_errors.append(np.abs(replay_weight - cached_weight))
            retained_count += int(source.size)

        query = response_start + np.arange(response_count, dtype=np.int64)
        diagonal_head = np.repeat(np.arange(head_count, dtype=np.int64), response_count)
        diagonal_query = np.tile(query, head_count)
        replay_diagonal = _selected_attention(
            channel,
            diagonal_head,
            diagonal_query,
            diagonal_query,
        ).reshape(head_count, response_count).T
        cached_layer_diagonal = cached_diagonal[:, layer]
        diagonal_errors.append(np.abs(replay_diagonal - cached_layer_diagonal))

        replay_known = replay_diagonal.copy()
        if source.size:
            np.add.at(
                replay_known,
                (target - response_start, head),
                replay_weight,
            )
        cached_known = 1.0 - cached_unresolved[:, layer]
        known_mass_errors.append(np.abs(replay_known - cached_known))

    retained_error = (
        np.concatenate(retained_errors)
        if retained_errors
        else np.empty(0, dtype=np.float64)
    )
    diagonal_error = np.concatenate([value.ravel() for value in diagonal_errors])
    known_error = np.concatenate([value.ravel() for value in known_mass_errors])
    retained_max = float(retained_error.max(initial=0.0))
    diagonal_max = float(diagonal_error.max(initial=0.0))
    known_max = float(known_error.max(initial=0.0))
    if max(retained_max, diagonal_max, known_max) > tolerance:
        raise ValueError(
            "replay attention does not match the frozen cache: "
            f"retained_max={retained_max:.6g}, "
            f"diagonal_max={diagonal_max:.6g}, "
            f"known_mass_max={known_max:.6g}, tolerance={tolerance:.6g}"
        )
    return AttentionCacheBinding(
        verified=True,
        absolute_tolerance=tolerance,
        retained_endpoints_compared=retained_count,
        diagonal_endpoints_compared=diagonal_count,
        retained_max_abs_error=retained_max,
        diagonal_max_abs_error=diagonal_max,
        known_mass_max_abs_error=known_max,
    )


__all__ = [
    "AttentionCacheBinding",
    "DEFAULT_ATTENTION_ATOL",
    "validate_replay_attention",
]
