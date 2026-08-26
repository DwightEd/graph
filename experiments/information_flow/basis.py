"""A shared low-dimensional source basis for graphs of different lengths."""

import torch
import torch.nn.functional as F


def fourier(values: torch.Tensor, dimensions: int) -> torch.Tensor:
    pairs = dimensions // 2
    if pairs == 0:
        return values.new_empty((len(values), 0))

    frequency = torch.arange(
        1,
        pairs + 1,
        device=values.device,
        dtype=values.dtype,
    )
    angle = torch.pi * values[:, None] * frequency[None]
    output = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
    if dimensions % 2:
        output = torch.cat((output, values[:, None]), dim=-1)
    return output


def source_basis(
    token_count: int,
    response_start: int,
    dimensions: int,
    device=None,
) -> torch.Tensor:
    """Encode role and position without using labels or token semantics."""

    position = torch.arange(token_count, device=device, dtype=torch.float32)
    absolute = position / max(token_count - 1, 1)
    response_count = token_count - response_start
    response = (position - response_start).clamp_min(0)
    response = response / max(response_count - 1, 1)

    role = torch.stack(
        (position < response_start, position >= response_start),
        dim=-1,
    ).float()
    role = role[:, : min(2, dimensions)]

    remaining = dimensions - role.shape[1]
    absolute_dimensions = remaining // 2
    response_dimensions = remaining - absolute_dimensions
    basis = torch.cat(
        (
            role,
            fourier(absolute, absolute_dimensions),
            fourier(response, response_dimensions),
        ),
        dim=-1,
    )
    return F.normalize(basis, dim=-1)
