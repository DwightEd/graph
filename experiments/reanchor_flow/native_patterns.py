"""Label-free joint modes of signed, projected native head and MLP writes.

All head coordinates enter one model. Layer scalars learned from the training
row bank preserve relative head amplitudes; no token-wise normalization or head
selection occurs. PCA and clustering describe recurring computation, not factual
correctness. Projection and PCA both lose information; reconstruction diagnostics
refer only to the supplied sketch, never to the complete residual space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd


@dataclass(frozen=True)
class NativePatternModel:
    """A frozen joint-write basis fitted to an externally source-balanced bank.

    Layout is head [layer, head, sketch] followed by MLP [layer, sketch]. Train
    RMS scalars are shared by all heads within a layer; MLP has its own layer
    scalar. These scalars change the discovery metric, not the native vectors.
    ``inverse_pattern_centers`` restores centroids to original sketch units.
    """

    head_shape: tuple[int, int, int]
    mlp_shape: tuple[int, int]
    scale: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    centers: np.ndarray
    explained_variance_ratio: np.ndarray
    fit_rows: int
    seed: int

    @staticmethod
    def vectorize(trace) -> np.ndarray:
        """Encode signed writes without using labels, logits or future rows."""
        head = np.asarray(trace["head_sketch"], dtype=np.float32)
        mlp = np.asarray(trace["mlp_sketch"], dtype=np.float32)
        if head.ndim != 4 or mlp.shape != (head.shape[0], *head.shape[2:]):
            raise ValueError("expected head [L,H,Q,R] and MLP [L,Q,R]")
        rows = head.shape[2]
        return np.concatenate(
            (np.moveaxis(head, 2, 0).reshape(rows, -1),
             np.moveaxis(mlp, 1, 0).reshape(rows, -1)), axis=1,
        )

    encode = vectorize

    @classmethod
    def fit(
        cls, matrix: np.ndarray, head_shape: tuple[int, int, int],
        mlp_shape: tuple[int, int], *, n_components: int = 8,
        n_patterns: int = 6, seed: int = 2026,
    ) -> NativePatternModel:
        """Fit only the caller's bounded, source-balanced TRAIN row bank.

        Returns a new model; it cannot mutate an already frozen fit. Pattern
        count is limited by distinct retained coordinates, including constant
        data. Zero training blocks use scale one rather than amplifying noise.
        """
        head_shape, mlp_shape = tuple(head_shape), tuple(mlp_shape)
        size = int(np.prod(head_shape))
        values = np.asarray(matrix, dtype=np.float32)
        if (len(head_shape) != 3 or mlp_shape != (head_shape[0], head_shape[2])
                or values.ndim != 2
                or values.shape[1] != size + int(np.prod(mlp_shape))
                or len(values) < 2 or min(n_components, n_patterns) < 1):
            raise ValueError("fit needs >=2 rows, matching write layout and positive sizes")
        if not np.isfinite(values).all():
            raise ValueError("training write sketches must be finite")
        head = values[:, :size].reshape(-1, *head_shape)
        mlp = values[:, size:].reshape(-1, *mlp_shape)
        head_rms = np.sqrt(np.mean(np.square(head), axis=(0, 2, 3), dtype=np.float64))
        mlp_rms = np.sqrt(np.mean(np.square(mlp), axis=(0, 2), dtype=np.float64))
        scale = np.concatenate((
            np.broadcast_to(head_rms[:, None, None], head_shape).ravel(),
            np.broadcast_to(mlp_rms[:, None], mlp_shape).ravel(),
        )).astype(np.float32)
        scale[scale == 0] = 1
        scaled = values / scale
        mean = scaled.mean(axis=0, dtype=np.float64).astype(np.float32)
        scaled -= mean
        count = min(n_components, len(values) - 1, values.shape[1])
        _, singular, components = randomized_svd(
            scaled, n_components=count, random_state=seed,
        )
        coords = scaled @ components.T
        clusters = min(n_patterns, len(np.unique(coords, axis=0)))
        centers = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(
            coords
        ).cluster_centers_
        energy = np.sum(np.square(scaled), dtype=np.float64)
        explained = singular.astype(np.float64) ** 2 / energy if energy else np.zeros(count)
        return cls(head_shape, mlp_shape, scale, mean, components, centers,
                   explained, len(values), seed)

    def transform(self, trace) -> dict[str, np.ndarray]:
        """Apply the frozen basis; novelty/transition are descriptive scores.

        ``distance`` is nearest-centroid Euclidean distance in retained PCA
        coordinates. ``reconstruction_error`` is RMS sketch error in scaled
        input units. Transition uses only the immediately preceding predictor
        row; the first row and gaps receive zero and ``has_previous=False``.
        """
        if (tuple(trace["head_sketch"].shape[:2]) + (trace["head_sketch"].shape[-1],)
                != self.head_shape):
            raise ValueError("trace geometry differs from fitted head layout")
        values = self.vectorize(trace) / self.scale - self.mean
        coords = values @ self.components.T
        distances = np.sum((coords[:, None] - self.centers[None]) ** 2, axis=-1)
        pattern = distances.argmin(axis=1)
        residual = values - coords @ self.components
        positions = np.asarray(trace["row_position"])
        has_previous = np.r_[False, np.diff(positions) == 1]
        transition = np.zeros(len(coords), dtype=np.float32)
        transition[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        transition[~has_previous] = 0
        return {
            "coords": coords, "pattern_id": pattern,
            "distance": np.sqrt(distances[np.arange(len(coords)), pattern]),
            "transition": transition, "has_previous": has_previous,
            "reconstruction_error": np.sqrt(np.mean(residual ** 2, axis=1)),
        }

    def _unpack(self, values: np.ndarray) -> dict[str, np.ndarray]:
        split = int(np.prod(self.head_shape))
        return {
            "head_sketch": values[:, :split].reshape(-1, *self.head_shape),
            "mlp_sketch": values[:, split:].reshape(-1, *self.mlp_shape),
        }

    def component_loadings(self) -> dict[str, np.ndarray]:
        """Signed PCA coordinate sensitivities to raw sketch values, per head."""
        return self._unpack(self.components / self.scale)

    def inverse_pattern_centers(self) -> dict[str, np.ndarray]:
        """Approximate pattern means, restoring native projected-write units."""
        return self._unpack((self.centers @ self.components + self.mean) * self.scale)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, pattern_schema=1, head_shape=self.head_shape, mlp_shape=self.mlp_shape,
            scale=self.scale, mean=self.mean, components=self.components,
            centers=self.centers, explained_variance_ratio=self.explained_variance_ratio,
            fit_rows=self.fit_rows, seed=self.seed,
        )

    @classmethod
    def load(cls, path: str | Path) -> NativePatternModel:
        with np.load(path, allow_pickle=False) as data:
            if int(data["pattern_schema"]) != 1:
                raise ValueError("unsupported native pattern schema")
            return cls(
                tuple(data["head_shape"].tolist()), tuple(data["mlp_shape"].tolist()),
                *(data[key] for key in ("scale", "mean", "components", "centers",
                                       "explained_variance_ratio")),
                int(data["fit_rows"]), int(data["seed"]),
            )
