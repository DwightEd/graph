"""One-class detection in learned GroundedRoute node embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .artifacts import load_npz, save_npz


REFERENCE_SCHEMA = "grounded-route-pca-knn-reference"
REFERENCE_VERSION = 1
MAD_SCALE = 1.482602218505602


@dataclass(frozen=True)
class PCAKNNConfig:
    components: int = 32
    neighbors: int = 20
    max_reference: int = 20_000
    seed: int = 20260825
    scale_floor: float = 1e-6


@dataclass(frozen=True)
class PCAWhitenedKNN:
    """Median/MAD normalization followed by whitened PCA and kNN distance."""

    median: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    whitening: np.ndarray
    reference: np.ndarray
    neighbors: int

    @property
    def collapsed(self) -> bool:
        return self.basis.shape[0] == 0

    @classmethod
    def fit(
        cls,
        embedding: np.ndarray,
        config: PCAKNNConfig | None = None,
    ) -> "PCAWhitenedKNN":
        config = PCAKNNConfig() if config is None else config
        values = embedding_matrix(embedding)

        if len(values) > config.max_reference:
            random = np.random.default_rng(config.seed)
            selected = random.choice(
                len(values),
                size=config.max_reference,
                replace=False,
            )
            values = values[np.sort(selected)]

        median = np.median(values, axis=0)
        mad = MAD_SCALE * np.median(np.abs(values - median), axis=0)
        scale = np.where(mad >= config.scale_floor, mad, 1.0)
        standardized = (values - median) / scale

        _, singular, right = np.linalg.svd(standardized, full_matrices=False)
        maximum = min(config.components, len(values) - 1, values.shape[1])
        threshold = config.scale_floor * np.sqrt(len(values) - 1)
        dimensions = min(maximum, int((singular > threshold).sum()))

        if dimensions == 0:
            warnings.warn(
                "Node embeddings collapsed to a constant reference; "
                "PCA-kNN will emit a constant zero score.",
                RuntimeWarning,
                stacklevel=2,
            )
            basis = np.empty((0, values.shape[1]), dtype=np.float32)
            whitening = np.empty(0, dtype=np.float32)
            reference = np.empty((len(values), 0), dtype=np.float32)
        else:
            basis = right[:dimensions].astype(np.float32)
            whitening = np.maximum(
                singular[:dimensions] / np.sqrt(len(values) - 1),
                config.scale_floor,
            ).astype(np.float32)
            reference = ((standardized @ basis.T) / whitening).astype(np.float32)

        return cls(
            median=median.astype(np.float32),
            scale=scale.astype(np.float32),
            basis=basis,
            whitening=whitening,
            reference=reference,
            neighbors=min(int(config.neighbors), len(values)),
        )

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        values = embedding_matrix(embedding)
        standardized = (values - self.median) / self.scale
        return ((standardized @ self.basis.T) / self.whitening).astype(np.float32)

    def score(self, embedding: np.ndarray) -> np.ndarray:
        """Return one mean-neighbor distance for every input node."""

        values = embedding_matrix(embedding)
        if self.collapsed:
            return np.zeros(len(values), dtype=np.float32)

        query = self.transform(values)
        neighbors = NearestNeighbors(
            n_neighbors=self.neighbors,
            metric="euclidean",
        ).fit(self.reference)
        distance, _ = neighbors.kneighbors(query, return_distance=True)
        return distance.mean(axis=1).astype(np.float32)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "median": self.median,
            "scale": self.scale,
            "basis": self.basis,
            "whitening": self.whitening,
            "reference": self.reference,
            "neighbors": np.asarray(self.neighbors, dtype=np.int32),
            "collapsed": np.asarray(self.collapsed),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "PCAWhitenedKNN":
        return cls(
            median=np.asarray(arrays["median"], dtype=np.float32),
            scale=np.asarray(arrays["scale"], dtype=np.float32),
            basis=np.asarray(arrays["basis"], dtype=np.float32),
            whitening=np.asarray(arrays["whitening"], dtype=np.float32),
            reference=np.asarray(arrays["reference"], dtype=np.float32),
            neighbors=int(np.asarray(arrays["neighbors"]).item()),
        )


def fit(
    embedding: np.ndarray,
    config: PCAKNNConfig | None = None,
) -> PCAWhitenedKNN:
    return PCAWhitenedKNN.fit(embedding, config)


def save_reference(path: str | Path, reference: PCAWhitenedKNN, **metadata) -> None:
    save_npz(
        path,
        schema=np.asarray(REFERENCE_SCHEMA),
        version=np.asarray(REFERENCE_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
        **reference.arrays(),
        **{name: np.asarray(value) for name, value in metadata.items()},
    )


def load_reference(path: str | Path) -> PCAWhitenedKNN:
    arrays = load_npz(path)
    return PCAWhitenedKNN.from_arrays(arrays)


def embedding_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("embedding must be a [node, dimension] matrix")
    return matrix
