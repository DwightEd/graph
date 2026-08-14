"""Role-preserving spectral representations of causal attention multiplexes.

The only data dependency is the canonical ``ResearchSample`` interface in
``research_dataset.py``.  This module never parses PT/NPZ cache files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd


REPRESENTATION_SCHEMA = "attention-dynamic-multiplex-spectral-v1"


@dataclass(frozen=True)
class MultiplexConfig:
    """Configuration for one sample's joint layer/head embedding."""

    rank: int = 16
    block_rows: int = 4096
    random_seed: int = 20260815
    include_diagonal: bool = True

    def validate(self) -> None:
        if int(self.rank) < 1:
            raise ValueError("rank must be positive")
        if int(self.block_rows) < 1:
            raise ValueError("block_rows must be positive")


@dataclass(frozen=True)
class MultiplexUnfolding:
    """Sparse ``(layer,target) x (head,source)`` attention unfolding."""

    mass_excess: sparse.csr_matrix
    shape_excess: sparse.csr_matrix
    layers: int
    heads: int
    response_tokens: int
    tokens: int
    response_idx: int
    attention_floor: float
    retained_off_diagonal_edges: int


@dataclass(frozen=True)
class SpectralRoles:
    """Query/source latent roles from one joint sparse SVD."""

    query_by_layer: np.ndarray
    source_by_head: np.ndarray
    singular_values: np.ndarray
    captured_energy: float


@dataclass(frozen=True)
class MultiplexRepresentation:
    """Two non-concatenated views of one attention multiplex.

    ``mass`` preserves margins above the cache floor. ``shape`` applies the
    square-root probability geometry before removing the same censoring
    baseline. Keeping the views separate prevents an arbitrary implicit
    weighting between them.
    """

    mass: SpectralRoles
    shape: SpectralRoles
    self_attention: np.ndarray
    retained_row_mass: np.ndarray
    unresolved_row_mass: np.ndarray
    response_idx: int
    token_count: int
    retained_off_diagonal_edges: int


def _as_numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else result


def build_multiplex_unfolding(
    sample,
    *,
    config: MultiplexConfig | None = None,
) -> MultiplexUnfolding:
    """Build a role-preserving sparse unfolding through the central data API.

    Let ``tau`` be ``attention_floor`` and let the central reconstructed view
    fill every censored causal edge by ``tau``.  The stored sparse matrices are
    the reconstructed matrices after subtracting that deterministic floor
    baseline:

    ``mass_excess = max(A - tau, 0)`` for retained off-diagonal edges,
    ``shape_excess = sqrt(A) - sqrt(tau)`` for those same edges.

    Censored edges are consequently implicit zeros rather than a huge set of
    artificial equal-weight edges. Exact diagonal values are stored directly;
    PP rows are absent because the cache cannot identify them.
    """

    config = MultiplexConfig() if config is None else config
    config.validate()
    attention = sample.attention()
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    response_tokens = int(attention.num_response_tokens)
    tokens = int(attention.num_tokens)
    response_idx = int(attention.response_idx)
    floor = float(attention.attention_floor)

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    shape_parts: list[np.ndarray] = []
    retained_edges = 0

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        if int(block.weight.numel()) == 0:
            continue
        layer = _as_numpy(block.layer, dtype=np.int64)
        head = _as_numpy(block.head, dtype=np.int64)
        query = _as_numpy(block.query, dtype=np.int64)
        source = _as_numpy(block.source, dtype=np.int64)
        weight = _as_numpy(block.weight, dtype=np.float64)
        row_parts.append(layer * response_tokens + query)
        column_parts.append(head * tokens + source)
        mass_parts.append(np.maximum(weight - floor, 0.0))
        shape_parts.append(
            np.maximum(np.sqrt(weight) - np.sqrt(floor), 0.0)
        )
        retained_edges += int(weight.size)

    if config.include_diagonal:
        query = np.arange(response_tokens, dtype=np.int64)
        target = response_idx + query
        diagonal = _as_numpy(
            attention.attention_diagonal[:, :, response_idx:], dtype=np.float64
        )
        layer_grid, head_grid, query_grid = np.meshgrid(
            np.arange(layers, dtype=np.int64),
            np.arange(heads, dtype=np.int64),
            query,
            indexing="ij",
        )
        row_parts.append(
            (layer_grid * response_tokens + query_grid).reshape(-1)
        )
        column_parts.append((head_grid * tokens + target[query_grid]).reshape(-1))
        mass_parts.append(diagonal.reshape(-1))
        shape_parts.append(np.sqrt(np.maximum(diagonal, 0.0)).reshape(-1))

    shape = (layers * response_tokens, heads * tokens)
    if not row_parts:
        empty = sparse.csr_matrix(shape, dtype=np.float32)
        return MultiplexUnfolding(
            mass_excess=empty,
            shape_excess=empty.copy(),
            layers=layers,
            heads=heads,
            response_tokens=response_tokens,
            tokens=tokens,
            response_idx=response_idx,
            attention_floor=floor,
            retained_off_diagonal_edges=retained_edges,
        )

    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    mass = np.concatenate(mass_parts).astype(np.float32, copy=False)
    probability_shape = np.concatenate(shape_parts).astype(np.float32, copy=False)
    mass_matrix = sparse.coo_matrix((mass, (rows, columns)), shape=shape).tocsr()
    shape_matrix = sparse.coo_matrix(
        (probability_shape, (rows, columns)), shape=shape
    ).tocsr()
    mass_matrix.eliminate_zeros()
    shape_matrix.eliminate_zeros()
    return MultiplexUnfolding(
        mass_excess=mass_matrix,
        shape_excess=shape_matrix,
        layers=layers,
        heads=heads,
        response_tokens=response_tokens,
        tokens=tokens,
        response_idx=response_idx,
        attention_floor=floor,
        retained_off_diagonal_edges=retained_edges,
    )


def _canonicalize_signs(left: np.ndarray, right: np.ndarray) -> None:
    """Choose deterministic component signs without changing the factorization."""

    for component in range(left.shape[1]):
        anchor = int(np.argmax(np.abs(left[:, component])))
        if left[anchor, component] < 0:
            left[:, component] *= -1
            right[:, component] *= -1


def joint_spectral_roles(
    matrix: sparse.csr_matrix,
    *,
    layers: int,
    heads: int,
    response_tokens: int,
    tokens: int,
    rank: int,
    random_seed: int,
) -> SpectralRoles:
    """Factor one unfolding into aligned query and source roles."""

    maximum_rank = min(matrix.shape)
    effective_rank = min(int(rank), maximum_rank)
    if effective_rank < 1:
        raise ValueError("unfolding has no valid spectral dimension")
    if matrix.nnz == 0:
        left = np.zeros((matrix.shape[0], effective_rank), dtype=np.float32)
        right = np.zeros((matrix.shape[1], effective_rank), dtype=np.float32)
        singular = np.zeros(effective_rank, dtype=np.float32)
    else:
        left, singular, right_transpose = randomized_svd(
            matrix,
            n_components=effective_rank,
            n_iter=5,
            random_state=int(random_seed),
            flip_sign=False,
        )
        right = right_transpose.T
        _canonicalize_signs(left, right)
        singular = singular.astype(np.float32, copy=False)
        scale = np.sqrt(np.maximum(singular, 0.0)).astype(np.float32)
        left = left.astype(np.float32, copy=False) * scale[None, :]
        right = right.astype(np.float32, copy=False) * scale[None, :]

    total_energy = float(np.square(matrix.data.astype(np.float64)).sum())
    captured = float(np.square(singular.astype(np.float64)).sum())
    captured_energy = captured / total_energy if total_energy > 0 else 0.0
    return SpectralRoles(
        query_by_layer=left.reshape(layers, response_tokens, effective_rank),
        source_by_head=right.reshape(heads, tokens, effective_rank),
        singular_values=singular,
        captured_energy=float(captured_energy),
    )


def represent_attention_multiplex(
    sample,
    *,
    config: MultiplexConfig | None = None,
) -> MultiplexRepresentation:
    """Create label-free token role trajectories for one attention sample."""

    config = MultiplexConfig() if config is None else config
    config.validate()
    unfolding = build_multiplex_unfolding(sample, config=config)
    mass = joint_spectral_roles(
        unfolding.mass_excess,
        layers=unfolding.layers,
        heads=unfolding.heads,
        response_tokens=unfolding.response_tokens,
        tokens=unfolding.tokens,
        rank=config.rank,
        random_seed=config.random_seed,
    )
    shape = joint_spectral_roles(
        unfolding.shape_excess,
        layers=unfolding.layers,
        heads=unfolding.heads,
        response_tokens=unfolding.response_tokens,
        tokens=unfolding.tokens,
        rank=config.rank,
        random_seed=config.random_seed,
    )

    attention = sample.attention()
    self_attention = _as_numpy(
        attention.attention_diagonal[:, :, attention.response_idx :],
        dtype=np.float32,
    )
    retained_mass = self_attention.copy()
    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        if int(block.weight.numel()) == 0:
            continue
        layer = _as_numpy(block.layer, dtype=np.int64)
        head = _as_numpy(block.head, dtype=np.int64)
        query = _as_numpy(block.query, dtype=np.int64)
        weight = _as_numpy(block.weight, dtype=np.float32)
        np.add.at(retained_mass, (layer, head, query), weight)
    unresolved = np.maximum(1.0 - retained_mass, 0.0).astype(np.float32)
    return MultiplexRepresentation(
        mass=mass,
        shape=shape,
        self_attention=self_attention,
        retained_row_mass=retained_mass,
        unresolved_row_mass=unresolved,
        response_idx=unfolding.response_idx,
        token_count=unfolding.tokens,
        retained_off_diagonal_edges=unfolding.retained_off_diagonal_edges,
    )
