"""Strictly bind exact replay attention to the canonical sparse cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class DenseSparseBinding:
    verified: bool
    absolute_tolerance: float
    retained_endpoints: int
    diagonal_endpoints: int
    retained_max_abs_error: float
    diagonal_max_abs_error: float
    censored_max_probability: float
    floor: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sample_fields(sample: Any) -> tuple[int, int, int, int, int]:
    required = (
        "num_layers",
        "num_heads",
        "num_tokens",
        "num_response_tokens",
        "response_idx",
    )
    if any(not hasattr(sample, name) for name in required):
        raise TypeError("expected a canonical AttentionSample")
    return tuple(int(getattr(sample, name)) for name in required)  # type: ignore[return-value]


def validate_exact_attention_against_cache(
    sample: Any,
    exact_response_attention: Sequence[torch.Tensor],
    *,
    absolute_tolerance: float,
) -> DenseSparseBinding:
    """Compare all measurable cache entries and all censored causal entries.

    Unlike a retained-endpoint-only check, the exact replay allows us to verify
    that every cache-omitted off-diagonal probability is genuinely at or below
    the cache floor.  A different checkpoint, head order, mask, or token order
    therefore fails before any graph artifact is created.
    """

    layers, heads, tokens, response, response_start = _sample_fields(sample)
    if len(exact_response_attention) != layers:
        raise ValueError("exact attention layer count differs from cache")
    tolerance = float(absolute_tolerance)
    floor = float(sample.attention_floor)
    if not 0.0 < tolerance < 0.1:
        raise ValueError("absolute_tolerance must lie in (0, 0.1)")

    retained_max = 0.0
    diagonal_max = 0.0
    censored_max = 0.0
    retained_count = 0
    diagonal_count = layers * heads * response
    row_ptr = sample.response_row_ptr.detach().cpu().long()
    columns = sample.response_column_indices.detach().cpu().long()
    weights = sample.response_values.detach().cpu().float()
    diagonal = sample.attention_diagonal.detach().cpu().float()
    if diagonal.shape != (layers, heads, tokens):
        raise ValueError("cached diagonal has invalid shape")

    source_index = torch.arange(tokens)
    for layer in range(layers):
        exact = torch.as_tensor(exact_response_attention[layer]).detach().cpu().float()
        if exact.shape != (heads, response, tokens):
            raise ValueError("exact attention must be [head,response,source]")
        observed = torch.zeros((heads, response, tokens), dtype=torch.bool)
        for head in range(heads):
            channel = layer * heads + head
            row_offset = channel * response
            for query in range(response):
                start = int(row_ptr[row_offset + query].item())
                stop = int(row_ptr[row_offset + query + 1].item())
                if stop > start:
                    selected_source = columns[start:stop]
                    selected_cache = weights[start:stop]
                    selected_exact = exact[head, query, selected_source]
                    retained_max = max(
                        retained_max,
                        float((selected_exact - selected_cache).abs().max().item()),
                    )
                    observed[head, query, selected_source] = True
                    retained_count += stop - start
                target = response_start + query
                cached_diagonal = diagonal[layer, head, target]
                exact_diagonal = exact[head, query, target]
                diagonal_max = max(
                    diagonal_max,
                    float((exact_diagonal - cached_diagonal).abs().item()),
                )
                observed[head, query, target] = True
                causal_off_diagonal = source_index < target
                censored = causal_off_diagonal & ~observed[head, query]
                if bool(censored.any()):
                    censored_max = max(
                        censored_max,
                        float(exact[head, query, censored].max().item()),
                    )

    if retained_max > tolerance or diagonal_max > tolerance:
        raise ValueError(
            "exact replay does not match cached retained/diagonal attention: "
            f"retained_max={retained_max:.6g}, diagonal_max={diagonal_max:.6g}, "
            f"tolerance={tolerance:.6g}"
        )
    if censored_max > floor + tolerance:
        raise ValueError(
            "cache omits an attention value above its declared floor: "
            f"censored_max={censored_max:.6g}, floor={floor:.6g}, "
            f"tolerance={tolerance:.6g}"
        )
    return DenseSparseBinding(
        verified=True,
        absolute_tolerance=tolerance,
        retained_endpoints=retained_count,
        diagonal_endpoints=diagonal_count,
        retained_max_abs_error=retained_max,
        diagonal_max_abs_error=diagonal_max,
        censored_max_probability=censored_max,
        floor=floor,
    )
