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


def _endpoint_description(
    name: str,
    endpoint: dict[str, int | float] | None,
) -> str:
    """Format one worst endpoint without hiding its absolute coordinates."""

    if endpoint is None:
        return f"worst_{name}=none"
    fields = [
        f"layer={int(endpoint['layer'])}",
        f"head={int(endpoint['head'])}",
        f"query={int(endpoint['query'])}",
    ]
    if "source" in endpoint:
        fields.append(f"source={int(endpoint['source'])}")
    fields.extend(
        (
            f"cache={float(endpoint['cache']):.9g}",
            f"replay={float(endpoint['replay']):.9g}",
            f"abs_error={float(endpoint['abs_error']):.9g}",
        )
    )
    return f"worst_{name}=({', '.join(fields)})"


def _per_layer_description(per_layer: Sequence[tuple[float, float, float]]) -> str:
    """Format retained/diagonal/known-mass maxima for every layer."""

    entries = [
        (
            f"L{layer}(retained={retained:.6g},diagonal={diagonal:.6g},"
            f"known_mass={known_mass:.6g})"
        )
        for layer, (retained, diagonal, known_mass) in enumerate(per_layer)
    ]
    return "per_layer_max=[" + "; ".join(entries) + "]"


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

    retained_count = 0
    diagonal_count = response_count * layer_count * head_count
    retained_max = 0.0
    diagonal_max = 0.0
    known_max = 0.0
    worst_retained: dict[str, int | float] | None = None
    worst_diagonal: dict[str, int | float] | None = None
    worst_known_mass: dict[str, int | float] | None = None
    per_layer: list[tuple[float, float, float]] = []

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
            retained_error = np.abs(replay_weight - cached_weight)
            retained_index = int(np.argmax(retained_error))
            layer_retained_max = float(retained_error[retained_index])
            if worst_retained is None or layer_retained_max > retained_max:
                retained_max = layer_retained_max
                worst_retained = {
                    "layer": layer,
                    "head": int(head[retained_index]),
                    "query": int(target[retained_index]),
                    "source": int(source[retained_index]),
                    "cache": float(cached_weight[retained_index]),
                    "replay": float(replay_weight[retained_index]),
                    "abs_error": layer_retained_max,
                }
            retained_count += int(source.size)
        else:
            replay_weight = np.empty(0, dtype=np.float64)
            layer_retained_max = 0.0

        query = response_start + np.arange(response_count, dtype=np.int64)
        diagonal_head = np.repeat(np.arange(head_count, dtype=np.int64), response_count)
        diagonal_query = np.tile(query, head_count)
        replay_diagonal = _selected_attention(
            channel,
            diagonal_head,
            diagonal_query,
            diagonal_query,
        ).reshape(head_count, response_count).T
        if not np.isfinite(replay_diagonal).all():
            raise ValueError("replay diagonal attention contains non-finite values")
        cached_layer_diagonal = cached_diagonal[:, layer]
        diagonal_error = np.abs(replay_diagonal - cached_layer_diagonal)
        diagonal_index = np.unravel_index(
            int(np.argmax(diagonal_error)), diagonal_error.shape
        )
        layer_diagonal_max = float(diagonal_error[diagonal_index])
        if worst_diagonal is None or layer_diagonal_max > diagonal_max:
            response_index, diagonal_head_index = map(int, diagonal_index)
            diagonal_max = layer_diagonal_max
            query_index = response_start + response_index
            worst_diagonal = {
                "layer": layer,
                "head": diagonal_head_index,
                "query": query_index,
                "source": query_index,
                "cache": float(cached_layer_diagonal[diagonal_index]),
                "replay": float(replay_diagonal[diagonal_index]),
                "abs_error": layer_diagonal_max,
            }

        replay_known = replay_diagonal.copy()
        if source.size:
            np.add.at(
                replay_known,
                (target - response_start, head),
                replay_weight,
            )
        cached_known = 1.0 - cached_unresolved[:, layer]
        known_error = np.abs(replay_known - cached_known)
        known_index = np.unravel_index(
            int(np.argmax(known_error)), known_error.shape
        )
        layer_known_max = float(known_error[known_index])
        if worst_known_mass is None or layer_known_max > known_max:
            response_index, known_head_index = map(int, known_index)
            known_max = layer_known_max
            worst_known_mass = {
                "layer": layer,
                "head": known_head_index,
                "query": response_start + response_index,
                "cache": float(cached_known[known_index]),
                "replay": float(replay_known[known_index]),
                "abs_error": layer_known_max,
            }
        per_layer.append(
            (layer_retained_max, layer_diagonal_max, layer_known_max)
        )

    if max(retained_max, diagonal_max, known_max) > tolerance:
        raise ValueError(
            "replay attention does not match the frozen cache: "
            f"retained_max={retained_max:.6g}, "
            f"diagonal_max={diagonal_max:.6g}, "
            f"known_mass_max={known_max:.6g}, tolerance={tolerance:.6g}; "
            f"{_endpoint_description('retained', worst_retained)}; "
            f"{_endpoint_description('diagonal', worst_diagonal)}; "
            f"{_endpoint_description('known_mass', worst_known_mass)}; "
            f"{_per_layer_description(per_layer)}"
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
