"""Permutation-invariant geometry of the per-layer attention-head point cloud."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig


@dataclass(frozen=True)
class HeadSetGeometry:
    """Layer-resolved geometry for every response token."""

    persistence_deaths: torch.Tensor  # [response, layer, head - 1]
    mean_death: torch.Tensor
    max_death: torch.Tensor
    persistence_entropy: torch.Tensor
    largest_gap: torch.Tensor
    effective_rank: torch.Tensor
    local_intrinsic_dimension: torch.Tensor
    layer_transition: torch.Tensor
    temporal_transition: torch.Tensor


def hellinger_distance_matrix(probability: torch.Tensor) -> torch.Tensor:
    """Pairwise Hellinger distance for ``[..., point, role]`` probabilities."""

    root = probability.clamp_min(0).sqrt()
    affinity = torch.einsum("...ik,...jk->...ij", root, root).clamp(0.0, 1.0)
    return (1.0 - affinity).clamp_min(0.0).sqrt()


def zero_dimensional_persistence(distance: torch.Tensor) -> torch.Tensor:
    """Return exact H0 death times via a batched Prim minimum spanning tree.

    For a finite point cloud, the finite death times of zero-dimensional
    persistent homology are exactly the edge weights of a minimum spanning tree.
    """

    batch_shape = distance.shape[:-2]
    points = int(distance.shape[-1])
    flat = distance.reshape(-1, points, points)
    if points <= 1:
        return torch.zeros((*batch_shape, 0), dtype=distance.dtype, device=distance.device)

    batches = flat.shape[0]
    row = torch.arange(batches, device=distance.device)
    visited = torch.zeros((batches, points), dtype=torch.bool, device=distance.device)
    visited[:, 0] = True
    minimum = flat[:, 0].clone()
    minimum[:, 0] = torch.inf
    deaths = []

    for _ in range(points - 1):
        chosen = minimum.argmin(dim=1)
        deaths.append(minimum[row, chosen])
        visited[row, chosen] = True
        minimum = torch.minimum(minimum, flat[row, chosen])
        minimum = minimum.masked_fill(visited, torch.inf)

    return torch.stack(deaths, dim=1).reshape(*batch_shape, points - 1)


def persistence_summaries(
    deaths: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if deaths.shape[-1] == 0:
        zeros = torch.zeros(deaths.shape[:-1], dtype=deaths.dtype, device=deaths.device)
        return zeros, zeros, zeros, zeros

    sorted_deaths = deaths.sort(dim=-1).values
    mean = sorted_deaths.mean(dim=-1)
    maximum = sorted_deaths.max(dim=-1).values
    total = sorted_deaths.sum(dim=-1)
    probability = sorted_deaths / total[..., None].clamp_min(epsilon)
    entropy = -(
        probability * probability.clamp_min(epsilon).log()
    ).sum(dim=-1)
    entropy = torch.where(total > epsilon, entropy, torch.zeros_like(entropy))

    if sorted_deaths.shape[-1] > 1:
        gap = sorted_deaths[..., 1:] - sorted_deaths[..., :-1]
        largest_gap = gap.max(dim=-1).values / maximum.clamp_min(epsilon)
        largest_gap = torch.where(
            maximum > epsilon, largest_gap, torch.zeros_like(largest_gap)
        )
    else:
        largest_gap = torch.zeros_like(maximum)
    return mean, maximum, entropy, largest_gap


def head_set_effective_rank(probability: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    """Spectral effective dimension of centered head-routing coordinates."""

    centered = probability - probability.mean(dim=-2, keepdim=True)
    covariance = torch.einsum("...hk,...hj->...kj", centered, centered)
    covariance = covariance / float(max(probability.shape[-2] - 1, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum(dim=-1)
    normalized = eigenvalues / total[..., None].clamp_min(epsilon)
    entropy = -(
        normalized * normalized.clamp_min(epsilon).log()
    ).sum(dim=-1)
    rank = entropy.exp()
    return torch.where(total > epsilon, rank, torch.zeros_like(rank))


def head_set_lid(
    distance: torch.Tensor,
    *,
    neighbors: int,
    epsilon: float,
) -> torch.Tensor:
    """Mean maximum-likelihood local intrinsic dimension across attention heads."""

    points = int(distance.shape[-1])
    keep = min(int(neighbors), points - 1)
    if keep < 2:
        return torch.zeros(distance.shape[:-2], dtype=distance.dtype, device=distance.device)

    diagonal = torch.eye(points, dtype=torch.bool, device=distance.device)
    expanded = diagonal.reshape(*((1,) * (distance.ndim - 2)), points, points)
    ordered = distance.masked_fill(expanded, torch.inf).sort(dim=-1).values[..., :keep]
    radius = ordered[..., -1]
    log_ratio = torch.log(ordered.clamp_min(epsilon) / radius[..., None].clamp_min(epsilon))
    denominator = log_ratio.mean(dim=-1)
    lid = torch.where(
        (radius > epsilon) & (denominator < -epsilon),
        -denominator.reciprocal(),
        torch.zeros_like(denominator),
    )
    return lid.mean(dim=-1)


def projection_directions(
    role_count: int,
    count: int,
    *,
    seed: int,
    device,
    dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = torch.randn((role_count, count), generator=generator, dtype=dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return directions.to(device=device)


def sliced_headset_transitions(
    probability: torch.Tensor,
    *,
    config: PhenomenologyConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation-invariant layer and token transitions via sliced Wasserstein."""

    directions = projection_directions(
        probability.shape[-1],
        config.transition_projections,
        seed=config.random_seed,
        device=probability.device,
        dtype=probability.dtype,
    )
    projected = torch.einsum("rlhk,kq->rlhq", probability, directions)
    ordered = projected.sort(dim=2).values

    layer = torch.zeros(
        probability.shape[:2], dtype=probability.dtype, device=probability.device
    )
    if probability.shape[1] > 1:
        layer[:, 1:] = (
            (ordered[:, 1:] - ordered[:, :-1]).square().mean(dim=(2, 3)).sqrt()
        )

    temporal = torch.zeros_like(layer)
    if probability.shape[0] > 1:
        temporal[1:] = (
            (ordered[1:] - ordered[:-1]).square().mean(dim=(2, 3)).sqrt()
        )
    return layer, temporal


def analyze_head_sets(
    role_probability: torch.Tensor,
    *,
    config: PhenomenologyConfig | None = None,
) -> HeadSetGeometry:
    config = PhenomenologyConfig() if config is None else config
    distance = hellinger_distance_matrix(role_probability)
    deaths = zero_dimensional_persistence(distance)
    mean, maximum, entropy, gap = persistence_summaries(
        deaths, epsilon=config.epsilon
    )
    effective_rank = head_set_effective_rank(
        role_probability, epsilon=config.epsilon
    )
    lid = head_set_lid(
        distance,
        neighbors=config.lid_neighbors,
        epsilon=config.epsilon,
    )
    layer_transition, temporal_transition = sliced_headset_transitions(
        role_probability, config=config
    )
    return HeadSetGeometry(
        persistence_deaths=deaths,
        mean_death=mean,
        max_death=maximum,
        persistence_entropy=entropy,
        largest_gap=gap,
        effective_rank=effective_rank,
        local_intrinsic_dimension=lid,
        layer_transition=layer_transition,
        temporal_transition=temporal_transition,
    )
