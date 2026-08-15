"""Label-free, blockwise one-class scoring for token representations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class OneClassConfig:
    """Fixed choices for one independently calibrated feature block."""

    position_bins: int = 10
    subspace_components: int = 32
    tail_fraction: float = 0.05
    seed: int = 42

    def validate(self) -> None:
        if int(self.position_bins) < 1:
            raise ValueError("position_bins must be positive")
        if int(self.subspace_components) < 1:
            raise ValueError("subspace_components must be positive")
        if not 0.0 < float(self.tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ScoreResult:
    """Unsupervised diagnostics and their independently calibrated score."""

    score: np.ndarray
    tail: np.ndarray
    subspace_residual: np.ndarray
    coordinates: np.ndarray


def _values(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or not len(result) or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _bins(bins: np.ndarray, rows: int, count: int) -> np.ndarray:
    result = np.asarray(bins)
    if result.ndim != 1 or len(result) != rows:
        raise ValueError("position bins must align with values")
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError("position bins must be integers")
    result = result.astype(np.int16, copy=False)
    if np.any(result < 0) or np.any(result >= count):
        raise ValueError("position bins are outside the configured range")
    return result


def _empirical_rank(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    return (
        (np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right") + .5)
        / (len(reference) + 1.0)
    ).astype(np.float32)


class OneClassReference:
    """One feature block's robust normal reference and independent ECDF tail."""

    def __init__(self, config: OneClassConfig):
        config.validate()
        self.config = config

    def fit(
        self,
        fit_values: np.ndarray,
        fit_bins: np.ndarray,
        cal_values: np.ndarray,
        cal_bins: np.ndarray,
    ) -> "OneClassReference":
        fit_values = _values(fit_values, name="fit_values")
        cal_values = _values(cal_values, name="cal_values")
        if fit_values.shape[1] != cal_values.shape[1]:
            raise ValueError("fit and calibration values must have the same width")
        if len(fit_values) < 2:
            raise ValueError("PCA fitting requires at least two fit rows")
        fit_bins = _bins(fit_bins, len(fit_values), self.config.position_bins)
        cal_bins = _bins(cal_bins, len(cal_values), self.config.position_bins)

        self.center, self.scale, self.bin_counts = self._fit_scaler(
            fit_values, fit_bins
        )
        standardized_fit = self._standardize_bins(fit_values, fit_bins)
        components = min(
            int(self.config.subspace_components), standardized_fit.shape[1],
            len(standardized_fit) - 1,
        )
        self.pca = PCA(
            n_components=max(1, components), svd_solver="randomized",
            random_state=int(self.config.seed),
        ).fit(standardized_fit)

        standardized_cal = self._standardize_bins(cal_values, cal_bins)
        tail, residual, _ = self._diagnostics(standardized_cal)
        self.tail_reference = np.sort(tail)
        self.residual_reference = np.sort(residual)
        combined = np.maximum(
            _empirical_rank(self.tail_reference, tail),
            _empirical_rank(self.residual_reference, residual),
        )
        self.combined_reference = np.sort(combined)
        self.fit_rows = len(fit_values)
        self.calibration_rows = len(cal_values)
        return self

    def transform(self, values: np.ndarray, position: np.ndarray) -> ScoreResult:
        self._require_fitted()
        values = _values(values, name="values")
        if values.shape[1] != self.center.shape[1]:
            raise ValueError("values do not match the fitted feature width")
        position = np.asarray(position, dtype=np.float64)
        if position.ndim != 1 or len(position) != len(values):
            raise ValueError("position must align with values")
        if not np.isfinite(position).all() or np.any(position < 0) or np.any(position > 1):
            raise ValueError("position must be finite and in [0, 1]")
        bins = np.minimum(
            (position * int(self.config.position_bins)).astype(np.int16),
            int(self.config.position_bins) - 1,
        )
        tail, residual, coordinates = self._diagnostics(
            self._standardize_bins(values, bins)
        )
        combined = np.maximum(
            _empirical_rank(self.tail_reference, tail),
            _empirical_rank(self.residual_reference, residual),
        )
        score = _empirical_rank(self.combined_reference, combined)
        return ScoreResult(score, tail, residual, coordinates)

    def state(self) -> dict[str, np.ndarray]:
        """Return only arrays needed to persist this label-free reference in NPZ."""
        self._require_fitted()
        return {
            "position_bins": np.asarray(self.config.position_bins, dtype=np.int16),
            "tail_fraction": np.asarray(self.config.tail_fraction, dtype=np.float32),
            "fit_rows": np.asarray(self.fit_rows, dtype=np.int32),
            "calibration_rows": np.asarray(self.calibration_rows, dtype=np.int32),
            "position_center": self.center.copy(),
            "position_scale": self.scale.copy(),
            "position_bin_counts": self.bin_counts.copy(),
            "pca_mean": self.pca.mean_.astype(np.float32, copy=True),
            "pca_components": self.pca.components_.astype(np.float32, copy=True),
            "pca_explained_variance": self.pca.explained_variance_.astype(
                np.float32, copy=True
            ),
            "tail_reference": self.tail_reference.copy(),
            "residual_reference": self.residual_reference.copy(),
            "combined_reference": self.combined_reference.copy(),
        }

    def _fit_scaler(
        self, values: np.ndarray, bins: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center, scale = self._statistics(values)
        centers = np.repeat(center[None, :], self.config.position_bins, axis=0)
        scales = np.repeat(scale[None, :], self.config.position_bins, axis=0)
        counts = np.bincount(bins, minlength=self.config.position_bins).astype(np.int32)
        for bin_id, count in enumerate(counts):
            if count >= 3:
                centers[bin_id], scales[bin_id] = self._statistics(values[bins == bin_id])
        return centers, scales, counts

    @staticmethod
    def _statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        std = values.std(axis=0)
        scale = np.where(mad > 1e-6, mad, np.where(std > 1e-6, std, 1.0))
        return center.astype(np.float32), scale.astype(np.float32)

    def _standardize_bins(self, values: np.ndarray, bins: np.ndarray) -> np.ndarray:
        standardized = (values - self.center[bins]) / self.scale[bins]
        return np.nan_to_num(
            standardized, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    def _diagnostics(self, standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        absolute = np.abs(standardized)
        keep = max(1, math.ceil(absolute.shape[1] * self.config.tail_fraction))
        tail = np.partition(absolute, absolute.shape[1] - keep, axis=1)[:, -keep:].mean(1)
        latent = self.pca.transform(standardized)
        reconstructed = self.pca.inverse_transform(latent)
        residual = np.mean((standardized - reconstructed) ** 2, axis=1)
        coordinates = np.zeros((len(standardized), 2), dtype=np.float32)
        coordinates[:, :min(2, latent.shape[1])] = latent[:, :2]
        return tail.astype(np.float32), residual.astype(np.float32), coordinates

    def _require_fitted(self) -> None:
        if not hasattr(self, "pca"):
            raise RuntimeError("fit must be called before transform or state")


class CalibratedMaxFusion:
    """Fuse independently calibrated blocks without changing their references."""

    def fit(self, calibrated_scores: dict[str, np.ndarray]) -> "CalibratedMaxFusion":
        maxima = self._maximum(calibrated_scores)
        self.block_names = tuple(sorted(calibrated_scores))
        self.reference = np.sort(maxima)
        return self

    def transform(self, calibrated_scores: dict[str, np.ndarray]) -> np.ndarray:
        if not hasattr(self, "reference"):
            raise RuntimeError("fit must be called before transform")
        if tuple(sorted(calibrated_scores)) != self.block_names:
            raise ValueError("fusion blocks do not match the fitted calibration blocks")
        return _empirical_rank(self.reference, self._maximum(calibrated_scores))

    def state(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "reference"):
            raise RuntimeError("fit must be called before state")
        return {
            "fusion_block_names": np.asarray(self.block_names),
            "fusion_reference": self.reference.copy(),
        }

    @staticmethod
    def _maximum(calibrated_scores: dict[str, np.ndarray]) -> np.ndarray:
        if not calibrated_scores:
            raise ValueError("fusion requires at least one calibrated score block")
        values = [np.asarray(calibrated_scores[name], dtype=np.float32) for name in sorted(calibrated_scores)]
        rows = len(values[0])
        if not rows or any(value.ndim != 1 or len(value) != rows for value in values):
            raise ValueError("calibrated score blocks must be non-empty aligned vectors")
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("calibrated scores must be finite")
        return np.maximum.reduce(values).astype(np.float32)
