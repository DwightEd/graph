"""Independent-validation effect maps for every layer and attention head."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HeadLayerEffects:
    feature_names: tuple[str, ...]
    positive_mean: np.ndarray
    negative_mean: np.ndarray
    standardized_mean_difference: np.ndarray
    positive_tokens: int
    negative_tokens: int

    def save(self, output_dir: Path, *, prefix: str) -> dict[str, Path]:
        """Write one complete numeric table, tensor artifact, and heatmap figure."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{prefix}_head_layer_effects.csv"
        tensor_path = output_dir / f"{prefix}_head_layer_effects.npz"
        figure_path = output_dir / f"{prefix}_head_layer_effects.png"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "layer",
                    "head",
                    "feature",
                    "positive_mean",
                    "negative_mean",
                    "standardized_mean_difference",
                    "positive_tokens",
                    "negative_tokens",
                ),
            )
            writer.writeheader()
            layers, heads, features = self.positive_mean.shape
            for feature in range(features):
                for layer in range(layers):
                    for head in range(heads):
                        writer.writerow(
                            {
                                "layer": layer,
                                "head": head,
                                "feature": self.feature_names[feature],
                                "positive_mean": float(
                                    self.positive_mean[layer, head, feature]
                                ),
                                "negative_mean": float(
                                    self.negative_mean[layer, head, feature]
                                ),
                                "standardized_mean_difference": float(
                                    self.standardized_mean_difference[
                                        layer, head, feature
                                    ]
                                ),
                                "positive_tokens": self.positive_tokens,
                                "negative_tokens": self.negative_tokens,
                            }
                        )

        np.savez_compressed(
            tensor_path,
            feature_names=np.asarray(self.feature_names, dtype=str),
            positive_mean=self.positive_mean.astype(np.float32),
            negative_mean=self.negative_mean.astype(np.float32),
            standardized_mean_difference=self.standardized_mean_difference.astype(
                np.float32
            ),
            positive_tokens=np.asarray(self.positive_tokens, dtype=np.int64),
            negative_tokens=np.asarray(self.negative_tokens, dtype=np.int64),
        )
        self._plot(figure_path)
        return {"csv": csv_path, "tensor": tensor_path, "figure": figure_path}

    def _plot(self, path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feature_count = len(self.feature_names)
        columns = min(4, feature_count)
        rows = (feature_count + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.5 * columns, 3.8 * rows),
            squeeze=False,
        )
        image = None
        for index, name in enumerate(self.feature_names):
            axis = axes.flat[index]
            image = axis.imshow(
                np.clip(self.standardized_mean_difference[..., index], -2.0, 2.0),
                aspect="auto",
                origin="lower",
                cmap="coolwarm",
                vmin=-2.0,
                vmax=2.0,
            )
            axis.set_title(name, fontsize=9)
            axis.set_xlabel("head")
            axis.set_ylabel("layer")
        for axis in axes.flat[feature_count:]:
            axis.axis("off")
        if image is not None:
            figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.5, label="SMD")
        figure.suptitle("Hallucination minus non-hallucination head/layer effects")
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)


class HeadLayerEffectMap:
    """Compute token-level standardized effects without collapsing head identity."""

    def compute(
        self,
        batches: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        feature_names: tuple[str, ...],
    ) -> HeadLayerEffects:
        positive_sum = None
        positive_square_sum = None
        negative_sum = None
        negative_square_sum = None
        positive_tokens = 0
        negative_tokens = 0

        for values, labels in batches:
            values = np.asarray(values, dtype=np.float64)
            labels = np.asarray(labels, dtype=np.int8)
            if values.ndim != 4 or values.shape[0] != len(labels):
                raise ValueError("effect-map batches must be [token, layer, head, feature]")
            if values.shape[-1] != len(feature_names):
                raise ValueError("effect-map feature names do not match tensor width")
            if not np.isfinite(values).all():
                raise FloatingPointError("effect-map features must be finite")
            if not np.isin(labels, (0, 1)).all():
                raise ValueError("effect-map labels must be binary")
            if positive_sum is None:
                shape = values.shape[1:]
                positive_sum = np.zeros(shape, dtype=np.float64)
                positive_square_sum = np.zeros(shape, dtype=np.float64)
                negative_sum = np.zeros(shape, dtype=np.float64)
                negative_square_sum = np.zeros(shape, dtype=np.float64)
            elif values.shape[1:] != positive_sum.shape:
                raise ValueError("effect-map geometry changes between batches")

            positive = values[labels == 1]
            negative = values[labels == 0]
            if len(positive):
                positive_sum += positive.sum(axis=0)
                positive_square_sum += np.square(positive).sum(axis=0)
                positive_tokens += len(positive)
            if len(negative):
                negative_sum += negative.sum(axis=0)
                negative_square_sum += np.square(negative).sum(axis=0)
                negative_tokens += len(negative)

        if positive_sum is None or positive_tokens < 2 or negative_tokens < 2:
            raise ValueError("effect map requires at least two tokens from each class")

        positive_mean = positive_sum / positive_tokens
        negative_mean = negative_sum / negative_tokens
        positive_variance = (
            positive_square_sum - positive_tokens * np.square(positive_mean)
        ) / (positive_tokens - 1)
        negative_variance = (
            negative_square_sum - negative_tokens * np.square(negative_mean)
        ) / (negative_tokens - 1)
        pooled_variance = (
            (positive_tokens - 1) * np.maximum(positive_variance, 0.0)
            + (negative_tokens - 1) * np.maximum(negative_variance, 0.0)
        ) / (positive_tokens + negative_tokens - 2)
        pooled_scale = np.sqrt(pooled_variance)
        difference = positive_mean - negative_mean
        standardized = np.divide(
            difference,
            pooled_scale,
            out=np.zeros_like(difference),
            where=pooled_scale > 1e-12,
        )
        return HeadLayerEffects(
            feature_names=feature_names,
            positive_mean=positive_mean.astype(np.float32),
            negative_mean=negative_mean.astype(np.float32),
            standardized_mean_difference=standardized.astype(np.float32),
            positive_tokens=positive_tokens,
            negative_tokens=negative_tokens,
        )
