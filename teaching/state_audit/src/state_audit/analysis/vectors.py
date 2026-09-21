"""General vector comparisons, with explicit group and pair axes."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def cosine(left, right):
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    result = np.full(denominator.shape, np.nan)
    return np.divide((left * right).sum(-1), denominator, out=result, where=denominator > 0)


@dataclass
class VectorComparison:
    groups: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    norms: np.ndarray
    cosines: np.ndarray


def compare_vectors(vectors: dict[str, np.ndarray]) -> VectorComparison:
    """Last axis is feature; preserve all preceding token/layer/head axes."""
    groups = tuple(vectors)
    pairs = tuple(combinations(groups, 2))
    norms = np.stack([np.linalg.norm(vectors[name], axis=-1) for name in groups], axis=-1)
    pair_cosines = [cosine(vectors[left], vectors[right]) for left, right in pairs]
    cosines = np.stack(pair_cosines, -1) if pairs else np.empty((*norms.shape[:-1], 0))
    return VectorComparison(groups, pairs, norms, cosines)
