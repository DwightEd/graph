"""One-class detection in learned GroundedRoute node embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    @classmethod
    def fit(
        cls,
        embedding: np.ndarray,
        config: PCAKNNConfig | None = None,
    ) -> "PCAWhitenedKNN":
        config = PCAKNNConfig() if config is None else config
        values = _embedding_matrix(embedding)
        if len(values) < 2:
            raise ValueError("PCA-kNN needs at least two reference embeddings")
        if config.components < 1 or config.neighbors < 1 or config.max_reference < 2:
            raise ValueError("PCA-kNN dimensions and reference sizes must be positive")

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
        if not dimensions:
            raise ValueError("PCA-kNN reference embeddings have no varying direction")
        basis = right[:dimensions]
        whitening = np.maximum(
            singular[:dimensions] / np.sqrt(len(values) - 1),
            config.scale_floor,
        )
        reference = (standardized @ basis.T) / whitening

        return cls(
            median=median.astype(np.float32),
            scale=scale.astype(np.float32),
            basis=basis.astype(np.float32),
            whitening=whitening.astype(np.float32),
            reference=reference.astype(np.float32),
            neighbors=min(int(config.neighbors), len(values)),
        )

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        values = _embedding_matrix(embedding)
        if values.shape[1] != len(self.median):
            raise ValueError("embedding dimension differs from the fitted reference")
        standardized = (values - self.median) / self.scale
        return ((standardized @ self.basis.T) / self.whitening).astype(np.float32)

    def score(self, embedding: np.ndarray) -> np.ndarray:
        """Return one mean-neighbor distance for every input node."""

        query = self.transform(embedding)
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
    if (
        str(arrays["schema"].item()) != REFERENCE_SCHEMA
        or int(arrays["version"].item()) != REFERENCE_VERSION
        or bool(arrays["labels_included"].item())
    ):
        raise ValueError("unsupported PCA-kNN reference artifact")
    return PCAWhitenedKNN.from_arrays(arrays)


def _embedding_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("embedding must be a finite [node, dimension] matrix")
    return matrix
