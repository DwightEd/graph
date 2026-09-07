"""Optional supervised readout of the scan, separate from routing anomalies.

This diagnostic uses correctness labels during fitting. Its matched position
control uses the same rows, label budget, weights and fixed optimizer settings.
Neither result is an unsupervised hallucination detector or a causal mechanism
test. Callers must restrict the fit factory to training sources.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .routing_transition import RoutingSequence


class _WeightedScaler:
    """Merge centered weighted moments; constant context stays zero variance."""

    def __init__(self) -> None:
        self.weight = 0.0

    def add(self, values: np.ndarray, row_weight: float) -> None:
        batch_weight = row_weight * len(values)
        # Center before summing so an exactly constant column has exactly zero
        # scatter, even when its value is not represented exactly in binary.
        batch_mean = values[0] + (values - values[0]).mean(axis=0)
        batch_m2 = row_weight * np.square(values - batch_mean).sum(axis=0)
        if self.weight == 0:
            self.mean_, self.m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - self.mean_
            total_weight = self.weight + batch_weight
            self.m2 += batch_m2 + delta**2 * self.weight * batch_weight / total_weight
            self.mean_ += delta * batch_weight / total_weight
        self.weight += batch_weight

    def finish(self) -> None:
        self.scale_ = np.sqrt(np.maximum(self.m2 / self.weight, 0))
        precision = np.finfo(np.float64).eps * np.maximum(1, np.abs(self.mean_))
        self.scale_[self.scale_ <= precision] = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.scale_


class SupervisedRoutingProbe:
    """Streaming, source-balanced linear diagnostics with no head averaging.

    Supply a fresh iterator factory yielding ``(sequence, labels)``. Labels are
    0 (nonhallucinated), 1 (hallucinated), or -1 (unknown). At most 128 evenly
    spaced known-label rows per sample are used. ``sequence.sample_weight``
    should be inverse source multiplicity, as in the unsupervised density fit.

    One pass fits source-balanced scalers; three passes fit fixed L2 logistic
    readouts. Both classes have equal total training weight. The routing probe
    receives each head's current state and adjacent-state difference, followed
    by causal context. The position probe receives only that context.
    """

    NAMES = ("supervised_routing", "supervised_position")
    EPOCHS = 3
    ALPHA = 1e-3
    SEED = 2026

    def __init__(self, *, max_rows_per_sample: int = 128) -> None:
        if max_rows_per_sample < 1:
            raise ValueError("max_rows_per_sample must be positive")
        self.max_rows_per_sample = max_rows_per_sample

    @staticmethod
    def _features(sequence: RoutingSequence, rows: np.ndarray) -> dict[str, np.ndarray]:
        current = sequence.state[rows].reshape(len(rows), -1)
        difference = np.zeros_like(current)
        adjacent = sequence.has_previous[rows]
        previous_rows = rows[adjacent] - 1
        difference[adjacent] = current[adjacent] - sequence.state[previous_rows].reshape(
            len(previous_rows), current.shape[1]
        )
        context = sequence.context[rows, 1:]
        return {
            "supervised_routing": np.concatenate((current, difference, context), axis=1),
            "supervised_position": context,
        }

    def _training_rows(self, sequence: RoutingSequence, labels: np.ndarray):
        labels = np.asarray(labels)
        if labels.shape != (len(sequence.state),) or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("probe labels must be one -1/0/1 value per scan row")
        eligible = np.flatnonzero(labels >= 0)
        if not len(eligible):
            return None
        rows = eligible[np.linspace(
            0, len(eligible) - 1,
            min(len(eligible), self.max_rows_per_sample), dtype=int,
        )]
        return rows, labels[rows].astype(np.int64), sequence.sample_weight / len(rows)

    def fit(
        self, factory: Callable[[], Iterable[tuple[RoutingSequence, np.ndarray]]]
    ) -> SupervisedRoutingProbe:
        # sklearn is needed only when the supervised diagnostic is requested.
        from sklearn.linear_model import SGDClassifier

        scalers = {name: _WeightedScaler() for name in self.NAMES}
        class_mass = np.zeros(2)
        class_count = np.zeros(2, dtype=np.int64)
        sample_count = 0
        for sequence, labels in factory():
            batch = self._training_rows(sequence, labels)
            if batch is None:
                continue
            rows, target, weight = batch
            self.state_shape = tuple(sequence.state.shape[1:])
            for name, values in self._features(sequence, rows).items():
                scalers[name].add(values, weight)
            counts = np.bincount(target, minlength=2)
            class_count += counts
            class_mass += weight * counts
            sample_count += 1
        if np.any(class_mass <= 0):
            raise ValueError("supervised diagnostic requires both known classes in fit sources")
        for scaler in scalers.values():
            scaler.finish()

        classifiers = {
            name: SGDClassifier(
                loss="log_loss", penalty="l2", alpha=self.ALPHA,
                random_state=self.SEED, average=True,
            )
            for name in self.NAMES
        }
        # Preserve source balance and equalize classes; normalize mean token
        # weight to one so SGD's fixed regularization has a stable scale.
        class_weight = class_count.sum() / (2 * class_mass)
        for _ in range(self.EPOCHS):
            for sequence, labels in factory():
                batch = self._training_rows(sequence, labels)
                if batch is None:
                    continue
                rows, target, weight = batch
                for name, values in self._features(sequence, rows).items():
                    classifiers[name].partial_fit(
                        scalers[name].transform(values), target,
                        classes=np.array([0, 1]), sample_weight=weight * class_weight[target],
                    )
        self.mean = {name: scalers[name].mean_ for name in self.NAMES}
        self.scale = {name: scalers[name].scale_ for name in self.NAMES}
        self.coefficient = {name: classifiers[name].coef_[0] for name in self.NAMES}
        self.intercept = {name: float(classifiers[name].intercept_[0]) for name in self.NAMES}
        self.fit_metadata = {
            "labels_used_for_fit": True,
            "interpretation": "supervised diagnostic; not the unsupervised detector",
            "fit_sample_count": sample_count,
            "fit_known_rows": int(class_count.sum()),
            "fit_class_rows": class_count.tolist(),
            "source_weighted_class_mass": class_mass.tolist(),
            "epochs": self.EPOCHS,
            "alpha": self.ALPHA,
            "seed": self.SEED,
            "class_balanced": True,
            "head_averaging": False,
            "max_rows_per_sample": self.max_rows_per_sample,
        }
        return self

    def score(self, sequence: RoutingSequence) -> dict[str, np.ndarray]:
        """Score every row; larger decision values predict hallucination.

        Bounded batches avoid making a full sample-by-feature matrix. Scores
        use only the current and adjacent previous row, never future length or
        future route states. A gap has a zero difference, just like the first
        row. Outputs are decision values, not calibrated probabilities.
        """
        if tuple(sequence.state.shape[1:]) != self.state_shape:
            raise ValueError("probe and scan layer/head state dimensions differ")
        output = {name: np.empty(len(sequence.state)) for name in self.NAMES}
        for start in range(0, len(sequence.state), 256):
            rows = np.arange(start, min(start + 256, len(sequence.state)))
            for name, values in self._features(sequence, rows).items():
                output[name][rows] = (
                    (values - self.mean[name]) / self.scale[name]
                ) @ self.coefficient[name] + self.intercept[name]
        return output

    def save(self, path: str | Path) -> None:
        """Save explicit numeric parameters; no pickle or executable payload."""
        arrays = {
            "probe_schema": np.array(1),
            "state_shape": np.asarray(self.state_shape),
            "fit_metadata": np.array(json.dumps(self.fit_metadata)),
        }
        for name in self.NAMES:
            for field in ("mean", "scale", "coefficient", "intercept"):
                arrays[f"{name}_{field}"] = np.asarray(getattr(self, field)[name])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> SupervisedRoutingProbe:
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays["probe_schema"]) != 1:
                raise ValueError("unsupported supervised probe schema")
            metadata = json.loads(str(arrays["fit_metadata"]))
            result = cls(max_rows_per_sample=metadata["max_rows_per_sample"])
            result.fit_metadata = metadata
            result.state_shape = tuple(arrays["state_shape"].tolist())
            for field in ("mean", "scale", "coefficient", "intercept"):
                setattr(result, field, {
                    name: arrays[f"{name}_{field}"].copy() for name in result.NAMES
                })
        return result
