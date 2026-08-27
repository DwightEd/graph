"""A shared low-dimensional source basis for graphs of different lengths."""

import torch
import torch.nn.functional as F


def fourier(values: torch.Tensor, dimensions: int) -> torch.Tensor:
    pairs = dimensions // 2
    if dimensions == 0:
        return values.new_empty((len(values), 0))
    if pairs:
        frequency = torch.arange(
            1,
            pairs + 1,
            device=values.device,
            dtype=values.dtype,
        )
        angle = torch.pi * values[:, None] * frequency[None]
        output = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
    else:
        output = values.new_empty((len(values), 0))
    if dimensions % 2:
        output = torch.cat((output, values[:, None]), dim=-1)
    return output


def source_basis(
    token_count: int,
    response_start: int,
    dimensions: int,
    device=None,
) -> torch.Tensor:
    """Encode role and boundary-relative position without total-length leakage."""

    position = torch.arange(token_count, device=device, dtype=torch.float32)
    prompt_role = position < response_start
    response_role = ~prompt_role
    role = torch.stack((prompt_role, response_role), dim=-1).float()
    role = role[:, : min(2, dimensions)]

    remaining = dimensions - role.shape[1]
    prompt_dimensions = remaining // 2
    response_dimensions = remaining - prompt_dimensions
    scale = torch.log(torch.tensor(4097.0, device=device))
    prompt_distance = torch.log1p(
        (response_start - 1 - position).clamp_min(0)
    ).div(scale).clamp_max(1.0)
    response_distance = torch.log1p(
        (position - response_start).clamp_min(0)
    ).div(scale).clamp_max(1.0)
    basis = torch.cat(
        (
            role,
            fourier(prompt_distance, prompt_dimensions)
            * prompt_role[:, None],
            fourier(response_distance, response_dimensions)
            * response_role[:, None],
        ),
        dim=-1,
    )
    return F.normalize(basis, dim=-1)
