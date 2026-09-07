"""A label-free conditional density of individual-head routing transitions.

Each head has three orthonormal log-ratio coordinates for the four source
buckets, plus log total message transport. A contextual Gaussian models the
previous/current pair, preserving the covariance between states. Heads are
independent factors: token NLL is their sum, never an averaged routing state.
This is a routing anomaly model, not a causal test of factual correctness.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scan_dataset import ScanSample

ILR_BASIS = (
    np.array([[1, -1, 0, 0], [1, 1, -2, 0], [1, 1, 1, -3]], dtype=np.float64)
    / np.sqrt(np.array([2, 6, 12]))[:, None]
)


@dataclass(frozen=True)
class RoutingSequence:
    """Causal state/context only; correctness labels are not part of this API."""

    state: np.ndarray  # [token, layer, head, 4]
    context: np.ndarray  # [token, 8]
    response_index: np.ndarray
    sample_weight: float = 1.0

    @classmethod
    def from_scan(
        cls, scan: ScanSample, *, sample_weight: float = 1.0, epsilon: float = 1e-5
    ) -> RoutingSequence:
        transport = np.moveaxis(
            np.asarray(scan["reanchor_bucket_transport"], dtype=np.float64), 2, 0
        )
        total = transport.sum(axis=-1, keepdims=True)
        fraction = np.divide(
            transport, total, out=np.zeros_like(transport), where=total > 0
        )
        log_fraction = np.log((fraction + epsilon) / (1 + 4 * epsilon))
        state = np.concatenate(
            (log_fraction @ ILR_BASIS.T, np.log(total + epsilon)), axis=-1
        )
        index = np.asarray(scan.response_index, dtype=np.int64)
        log_position = np.log1p(index) / 5
        log_prompt = np.full(len(index), np.log1p(scan.response_start) / 8)
        context = np.column_stack(
            (
                np.ones(len(index)),
                log_position,
                log_position**2,
                log_prompt,
                log_prompt**2,
                log_position * log_prompt,
                index > 0,
                index > int(scan.metadata["local_window"]) + 1,
            )
        )
        return cls(state, context, index, sample_weight)

    @property
    def has_previous(self) -> np.ndarray:
        return np.r_[False, np.diff(self.response_index) == 1]


@dataclass(frozen=True)
class RoutingScores:
    joint: np.ndarray
    current: np.ndarray
    previous: np.ndarray
    innovation: np.ndarray
    per_head_joint: np.ndarray
    per_head_current: np.ndarray
    per_head_previous: np.ndarray
    per_head_innovation: np.ndarray
    has_previous: np.ndarray


@dataclass
class _RegressionMoments:
    xtx: np.ndarray
    xty: np.ndarray
    weight: float = 0.0

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> _RegressionMoments:
        return cls(np.zeros((8, 8)), np.zeros((8, *shape)))

    def add(self, x: np.ndarray, y: np.ndarray, weight: float) -> None:
        self.xtx += weight * (x.T @ x)
        self.xty += weight * np.einsum("tc,tlhd->clhd", x, y, optimize=True)
        self.weight += weight * len(x)

    def solve(self, ridge: float) -> np.ndarray:
        regularizer = np.eye(8)
        regularizer[0, 0] = 0
        matrix = self.xtx / self.weight + ridge * regularizer
        solution = np.linalg.solve(matrix, self.xty.reshape(8, -1) / self.weight)
        return solution.reshape(self.xty.shape)


def _mean(context: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    return np.einsum("tc,clhd->tlhd", context, coefficient, optimize=True)


def _precision(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.linalg.inv(covariance), np.linalg.slogdet(covariance)[1]


def _nll(residual: np.ndarray, precision: np.ndarray, logdet: np.ndarray) -> np.ndarray:
    energy = np.einsum(
        "tlhi,lhij,tlhj->tlh", residual, precision, residual, optimize=True
    )
    return 0.5 * (energy + logdet + residual.shape[-1] * np.log(2 * np.pi))


class RoutingTransitionModel:
    """Two-pass streaming fit: ridge mean, then shrunk residual covariance.

    Give ``fit`` a fresh iterator factory because it reads each sequence twice.
    Each sequence contributes its sample weight, independent of response length;
    use inverse source multiplicity for equal source weights. At most ``max_rows``
    deterministic evenly spaced rows per sequence enter each density fit.

    The current-state control is fitted on all rows, including the first row.
    The joint density is fitted on adjacent pairs. Its exact Gaussian chain-rule
    decomposition is previous-state NLL plus conditional-innovation NLL.
    """

    def __init__(
        self,
        *,
        ridge: float = 1e-3,
        shrinkage: float = 0.05,
        max_rows_per_sample: int = 128,
        epsilon: float = 1e-5,
    ) -> None:
        if ridge <= 0 or not 0 < shrinkage <= 1 or max_rows_per_sample < 1:
            raise ValueError("ridge/max_rows must be positive; shrinkage in (0, 1]")
        self.ridge = ridge
        self.shrinkage = shrinkage
        self.max_rows_per_sample = max_rows_per_sample
        self.epsilon = epsilon

    def _observations(self, sequence: RoutingSequence):
        for name, eligible in (
            ("current", np.arange(len(sequence.state))),
            ("joint", np.flatnonzero(sequence.has_previous)),
        ):
            if not len(eligible):
                continue
            chosen = eligible[
                np.linspace(
                    0,
                    len(eligible) - 1,
                    min(len(eligible), self.max_rows_per_sample),
                    dtype=int,
                )
            ]
            value = sequence.state[chosen]
            if name == "joint":
                value = np.concatenate((sequence.state[chosen - 1], value), axis=-1)
            yield (
                name,
                sequence.context[chosen],
                value,
                sequence.sample_weight / len(chosen),
            )

    def fit(
        self, factory: Callable[[], Iterable[RoutingSequence]]
    ) -> RoutingTransitionModel:
        moments: dict[str, _RegressionMoments] = {}
        for sequence in factory():
            for name, context, value, weight in self._observations(sequence):
                if name not in moments:
                    moments[name] = _RegressionMoments.empty(value.shape[1:])
                if value.shape[1:] != moments[name].xty.shape[1:]:
                    raise ValueError("training scans disagree on layer/head geometry")
                moments[name].add(context, value, weight)
        if set(moments) != {"current", "joint"}:
            raise ValueError("fit requires nonempty sequences with adjacent rows")
        self.coefficients = {
            name: item.solve(self.ridge) for name, item in moments.items()
        }
        covariance = {
            name: np.zeros(
                (*coefficient.shape[1:3], coefficient.shape[-1], coefficient.shape[-1])
            )
            for name, coefficient in self.coefficients.items()
        }
        for sequence in factory():
            for name, context, value, weight in self._observations(sequence):
                residual = value - _mean(context, self.coefficients[name])
                covariance[name] += weight * np.einsum(
                    "tlhi,tlhj->lhij", residual, residual, optimize=True
                )
        self.covariances = {}
        for name, value in covariance.items():
            value /= moments[name].weight
            dimension = value.shape[-1]
            diagonal = np.diagonal(value, axis1=-2, axis2=-1)
            identity = np.eye(dimension)
            floor = np.maximum(diagonal.mean(axis=-1) * 1e-6, 1e-7)
            self.covariances[name] = (
                (1 - self.shrinkage) * value
                + self.shrinkage * identity * diagonal[..., None, :]
                + identity * floor[..., None, None]
            )
        self._prepare()
        return self

    def _prepare(self) -> None:
        self.current_precision, self.current_logdet = _precision(
            self.covariances["current"]
        )
        joint = self.covariances["joint"]
        self.previous_precision, self.previous_logdet = _precision(joint[..., :4, :4])
        self.conditional_gain = joint[..., 4:, :4] @ self.previous_precision
        self.conditional_covariance = (
            joint[..., 4:, 4:] - self.conditional_gain @ joint[..., :4, 4:]
        )
        self.innovation_precision, self.innovation_logdet = _precision(
            self.conditional_covariance
        )

    def score(self, sequence: RoutingSequence) -> RoutingScores:
        state = sequence.state
        if state.shape[1:] != self.coefficients["current"].shape[1:]:
            raise ValueError("scan layer/head geometry differs from fitted model")
        current = _nll(
            state - _mean(sequence.context, self.coefficients["current"]),
            self.current_precision,
            self.current_logdet,
        )
        previous = np.full_like(current, np.nan)
        innovation = np.full_like(current, np.nan)
        has_previous = sequence.has_previous
        rows = np.flatnonzero(has_previous)
        expected = _mean(sequence.context[rows], self.coefficients["joint"])
        residual_previous = state[rows - 1] - expected[..., :4]
        residual_current = state[rows] - expected[..., 4:]
        residual_innovation = residual_current - np.einsum(
            "lhij,tlhj->tlhi", self.conditional_gain, residual_previous, optimize=True
        )
        previous[rows] = _nll(
            residual_previous, self.previous_precision, self.previous_logdet
        )
        innovation[rows] = _nll(
            residual_innovation, self.innovation_precision, self.innovation_logdet
        )
        joint = current.copy()
        joint[rows] = previous[rows] + innovation[rows]
        return RoutingScores(
            *(
                value.sum(axis=(1, 2))
                for value in (joint, current, previous, innovation)
            ),
            joint,
            current,
            previous,
            innovation,
            has_previous,
        )

    def head_excess(self, scores: RoutingScores) -> np.ndarray:
        """Half Mahalanobis distance minus its model expectation, per head.

        Raw head NLL includes a covariance-volume offset, so a diffuse head can
        have large NLL without being unusual. Subtracting each fitted Gaussian's
        expected NLL removes that offset. Use this quantity to rank explanatory
        heads; the token detector still uses the exact likelihood above.
        """
        log_normalizer = np.log(2 * np.pi) + 1
        current_expected = 0.5 * (self.current_logdet + 4 * log_normalizer)
        joint_expected = 0.5 * (
            self.previous_logdet + self.innovation_logdet + 8 * log_normalizer
        )
        excess = scores.per_head_joint - current_expected
        excess[scores.has_previous] = (
            scores.per_head_joint[scores.has_previous] - joint_expected
        )
        return excess

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            schema=np.array(1),
            ridge=self.ridge,
            shrinkage=self.shrinkage,
            max_rows_per_sample=self.max_rows_per_sample,
            epsilon=self.epsilon,
            **{f"coefficient_{key}": value for key, value in self.coefficients.items()},
            **{f"covariance_{key}": value for key, value in self.covariances.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> RoutingTransitionModel:
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["schema"]) != 1:
                raise ValueError("unsupported routing transition schema")
            model = cls(
                ridge=float(saved["ridge"]),
                shrinkage=float(saved["shrinkage"]),
                max_rows_per_sample=int(saved["max_rows_per_sample"]),
                epsilon=float(saved["epsilon"]),
            )
            model.coefficients = {
                key: saved[f"coefficient_{key}"] for key in ("current", "joint")
            }
            model.covariances = {
                key: saved[f"covariance_{key}"] for key in ("current", "joint")
            }
        model._prepare()
        return model
