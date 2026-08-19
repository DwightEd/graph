"""Permutation-invariant geometry of per-layer attention-head point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig


@dataclass(frozen=True)
class HeadSetGeometry:
    persistence_deaths: torch.Tensor
    mean_death: torch.Tensor
    max_death: torch.Tensor
    persistence_entropy: torch.Tensor
    largest_gap: torch.Tensor
    effective_rank: torch.Tensor
    local_intrinsic_dimension: torch.Tensor
    layer_transition: torch.Tensor
    temporal_transition: torch.Tensor


def hellinger_distance_matrix(probability: torch.Tensor) -> torch.Tensor:
    """Pairwise Hellinger distances for ``[..., point, role]`` probabilities."""

    root = probability.clamp_min(0.0).sqrt()
    affinity = torch.einsum("...ik,...jk->...ij", root, root).clamp(0.0, 1.0)
    return (1.0 - affinity).sqrt()


def zero_dimensional_persistence(distance: torch.Tensor) -> torch.Tensor:
    """Compute finite H0 death times as minimum-spanning-tree edge weights."""

    batch_shape = distance.shape[:-2]
    points = distance.shape[-1]
    if points <= 1:
        return distance.new_zeros((*batch_shape, 0))

    flat = distance.reshape(-1, points, points)
    rows = torch.arange(len(flat), device=distance.device)
    visited = torch.zeros((len(flat), points), dtype=torch.bool, device=distance.device)
    visited[:, 0] = True
    nearest = flat[:, 0].clone()
    nearest[:, 0] = torch.inf
    deaths = []

    for _ in range(points - 1):
        chosen = nearest.argmin(dim=1)
        deaths.append(nearest[rows, chosen])
        visited[rows, chosen] = True
        nearest = torch.minimum(nearest, flat[rows, chosen])
        nearest = nearest.masked_fill(visited, torch.inf)

    return torch.stack(deaths, dim=1).reshape(*batch_shape, points - 1)


def persistence_summaries(
    deaths: torch.Tensor, *, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if deaths.shape[-1] == 0:
        zero = deaths.new_zeros(deaths.shape[:-1])
        return zero, zero, zero, zero

    ordered = deaths.sort(dim=-1).values
    mean = ordered.mean(dim=-1)
    maximum = ordered[..., -1]
    total = ordered.sum(dim=-1)
    probability = ordered / total[..., None].clamp_min(epsilon)
    entropy = -(probability * probability.clamp_min(epsilon).log()).sum(dim=-1)
    entropy = torch.where(total > epsilon, entropy, torch.zeros_like(entropy))

    if ordered.shape[-1] == 1:
        gap = torch.zeros_like(maximum)
    else:
        gap = (ordered[..., 1:] - ordered[..., :-1]).max(dim=-1).values
        gap = torch.where(maximum > epsilon, gap / maximum, torch.zeros_like(gap))
    return mean, maximum, entropy, gap


def head_set_effective_rank(probability: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    centered = probability - probability.mean(dim=-2, keepdim=True)
    covariance = torch.einsum("...hk,...hj->...kj", centered, centered)
    covariance /= float(max(probability.shape[-2] - 1, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum(dim=-1)
    normalized = eigenvalues / total[..., None].clamp_min(epsilon)
    entropy = -(normalized * normalized.clamp_min(epsilon).log()).sum(dim=-1)
    rank = entropy.exp()
    return torch.where(total > epsilon, rank, torch.zeros_like(rank))


def head_set_lid(
    distance: torch.Tensor, *, neighbors: int, epsilon: float
) -> torch.Tensor:
    """Mean maximum-likelihood local intrinsic dimension across heads."""

    points = distance.shape[-1]
    keep = min(neighbors, points - 1)
    if keep < 2:
        return distance.new_zeros(distance.shape[:-2])

    diagonal = torch.eye(points, dtype=torch.bool, device=distance.device)
    diagonal = diagonal.reshape(*((1,) * (distance.ndim - 2)), points, points)
    neighbors_distance = distance.masked_fill(diagonal, torch.inf).sort(dim=-1).values
    neighbors_distance = neighbors_distance[..., :keep]
    radius = neighbors_distance[..., -1]
    log_ratio = torch.log(
        neighbors_distance.clamp_min(epsilon) / radius[..., None].clamp_min(epsilon)
    )
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
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn((role_count, count), generator=generator, dtype=dtype)
    directions /= directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return directions.to(device=device)


def sliced_headset_transitions(
    probability: torch.Tensor, *, config: PhenomenologyConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare unordered head sets across layer depth and token time."""

    directions = projection_directions(
        probability.shape[-1],
        config.transition_projections,
        seed=config.random_seed,
        device=probability.device,
        dtype=probability.dtype,
    )
    projected = torch.einsum("tlhr,rq->tlhq", probability, directions)
    ordered = projected.sort(dim=2).values

    layer = probability.new_zeros(probability.shape[:2])
    layer[:, 1:] = (
        (ordered[:, 1:] - ordered[:, :-1]).square().mean(dim=(2, 3)).sqrt()
    )
    temporal = probability.new_zeros(probability.shape[:2])
    temporal[1:] = (
        (ordered[1:] - ordered[:-1]).square().mean(dim=(2, 3)).sqrt()
    )
    return layer, temporal


def analyze_head_sets(
    probability: torch.Tensor,
    *,
    config: PhenomenologyConfig | None = None,
) -> HeadSetGeometry:
    config = PhenomenologyConfig() if config is None else config
    distance = hellinger_distance_matrix(probability)
    deaths = zero_dimensional_persistence(distance)
    mean, maximum, entropy, gap = persistence_summaries(
        deaths, epsilon=config.epsilon
    )
    layer_transition, temporal_transition = sliced_headset_transitions(
        probability, config=config
    )
    return HeadSetGeometry(
        persistence_deaths=deaths,
        mean_death=mean,
        max_death=maximum,
        persistence_entropy=entropy,
        largest_gap=gap,
        effective_rank=head_set_effective_rank(probability, epsilon=config.epsilon),
        local_intrinsic_dimension=head_set_lid(
            distance, neighbors=config.lid_neighbors, epsilon=config.epsilon
        ),
        layer_transition=layer_transition,
        temporal_transition=temporal_transition,
    )
