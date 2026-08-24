"""Label-free, finite-sample calibration for channel-wise anomaly scores.

The calibration contract deliberately separates two empirical references.  The
first reference calibrates each layer/head channel independently.  An optional
second reference is transformed through those frozen channel ECDFs and is used
to calibrate the resulting Cauchy fusion statistic.  This prevents a token from
being compared with a fusion distribution that contains the token itself.

When no independent fusion reference is supplied, leave-one-out channel
``p``-values provide a conservative single-stream fallback.  That fallback is
useful for small experiments and tests, but production runs should reserve
disjoint source groups for the channel and fusion references.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


class AlignedFloat32Reservoir:
    """Keep aligned bottom-k rows without quantizing unbounded phase scores.

    The repository's older aligned reservoir stores every block as float16,
    which is appropriate for bounded representations but can overflow a CUSUM
    score on a long response. This experiment-local reservoir has one global
    stratum, keeps all blocks at identical row slots, and stores float32.
    """

    def __init__(self, *, size: int, seed: int) -> None:
        if isinstance(size, bool) or int(size) < 2:
            raise ValueError("reservoir size must be at least two")
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("reservoir seed must be non-negative")
        self.capacity = int(size)
        self._rng = np.random.default_rng(int(seed))
        self._priorities = np.empty(0, dtype=np.float64)
        self._values: dict[str, np.ndarray] | None = None
        self._widths: dict[str, int] | None = None

    def add(self, blocks: Mapping[str, np.ndarray]) -> None:
        values = {
            str(name): _float_matrix(
                value,
                name=f"reservoir block {name}",
                minimum_rows=1,
            )
            for name, value in blocks.items()
        }
        if not values:
            raise ValueError("reservoir blocks cannot be empty")
        rows = len(next(iter(values.values())))
        if any(len(value) != rows for value in values.values()):
            raise ValueError("reservoir blocks must share token rows")
        widths = {name: int(value.shape[1]) for name, value in values.items()}
        if self._values is None:
            self._widths = widths
            self._values = {
                name: np.empty((0, width), dtype=np.float32)
                for name, width in widths.items()
            }
        elif widths != self._widths or tuple(values) != tuple(self._values):
            raise ValueError("reservoir block contract changed after first add")

        priorities = self._rng.random(rows)
        candidates = np.concatenate((self._priorities, priorities))
        keep = np.argsort(candidates, kind="stable")[: self.capacity]
        assert self._values is not None
        for name in self._values:
            combined = np.concatenate((self._values[name], values[name]), axis=0)
            self._values[name] = combined[keep].astype(np.float32, copy=False)
        self._priorities = candidates[keep]

    def values(self) -> dict[str, np.ndarray]:
        if self._values is None or len(self._priorities) < 2:
            raise ValueError("reservoir contains fewer than two rows")
        return {name: value.copy() for name, value in self._values.items()}


def _float_matrix(
    values,
    *,
    name: str,
    minimum_rows: int,
    columns: int | None = None,
) -> np.ndarray:
    """Return one finite two-dimensional matrix in computation precision."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [rows, channels]")
    if len(array) < int(minimum_rows):
        raise ValueError(f"{name} must contain at least {minimum_rows} rows")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    if columns is not None and array.shape[1] != int(columns):
        raise ValueError(f"{name} has a different channel count")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must use a numeric dtype")
    computed = np.asarray(array, dtype=np.float32)
    if not bool(np.isfinite(computed).all()):
        raise ValueError(f"{name} contains non-finite values")
    return computed


def _float_vector(values, *, name: str, minimum_rows: int) -> np.ndarray:
    """Return one finite one-dimensional vector in computation precision."""

    array = np.asarray(values)
    if array.ndim != 1 or len(array) < int(minimum_rows):
        raise ValueError(
            f"{name} must be a vector with at least {minimum_rows} entries"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must use a numeric dtype")
    computed = np.asarray(array, dtype=np.float32)
    if not bool(np.isfinite(computed).all()):
        raise ValueError(f"{name} contains non-finite values")
    return computed


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    """Frozen ECDF references used by :func:`score_channels`.

    Parameters
    ----------
    calibration_channel_score:
        Matrix with shape ``[K, C]``.  It may be stored as ``float16``; every
        numerical operation promotes it to ``float32``.
    calibration_fusion_stat:
        Vector with shape ``[K_fusion]`` containing Cauchy statistics computed
        either from an independent reference or by leave-one-out fallback.
    independent_fusion_reference:
        Records which construction was used.  It is an audit fact, not a model
        parameter.
    """

    calibration_channel_score: np.ndarray
    calibration_fusion_stat: np.ndarray
    independent_fusion_reference: bool = False
    _sorted_channel_score: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )
    _sorted_fusion_stat: np.ndarray = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        channel = np.asarray(self.calibration_channel_score)
        if channel.ndim != 2 or len(channel) < 2 or channel.shape[1] < 1:
            raise ValueError(
                "calibration_channel_score must have shape [K>=2, C>=1]"
            )
        if not np.issubdtype(channel.dtype, np.floating):
            raise ValueError("calibration_channel_score must use a floating dtype")
        if not bool(np.isfinite(channel).all()):
            raise ValueError("calibration_channel_score contains non-finite values")

        fusion = np.asarray(self.calibration_fusion_stat)
        if fusion.ndim != 1 or len(fusion) < 2:
            raise ValueError(
                "calibration_fusion_stat must contain at least two entries"
            )
        if not np.issubdtype(fusion.dtype, np.floating):
            raise ValueError("calibration_fusion_stat must use a floating dtype")
        if not bool(np.isfinite(fusion).all()):
            raise ValueError("calibration_fusion_stat contains non-finite values")

        # Own immutable copies so a caller cannot silently mutate a fitted
        # reference after it has been validated or written to an artifact.
        channel_copy = np.array(channel, copy=True)
        fusion_copy = np.array(fusion, dtype=np.float32, copy=True)
        # ECDF scoring always promotes the persisted channel matrix to float32
        # before sorting.  Cache exactly that computation here: the persisted
        # arrays and artifact schema remain unchanged, while every scored
        # sample reuses one immutable ordering instead of sorting all channels
        # again.  Sorting the fusion vector once provides the same guarantee
        # for the second calibration stage.
        sorted_channel = np.asfortranarray(
            np.sort(
                channel_copy.astype(np.float32, copy=False),
                axis=0,
            )
        )
        sorted_fusion = np.sort(fusion_copy)
        channel_copy.setflags(write=False)
        fusion_copy.setflags(write=False)
        sorted_channel.setflags(write=False)
        sorted_fusion.setflags(write=False)
        object.__setattr__(self, "calibration_channel_score", channel_copy)
        object.__setattr__(self, "calibration_fusion_stat", fusion_copy)
        object.__setattr__(self, "_sorted_channel_score", sorted_channel)
        object.__setattr__(self, "_sorted_fusion_stat", sorted_fusion)
        object.__setattr__(
            self,
            "independent_fusion_reference",
            bool(self.independent_fusion_reference),
        )

    @property
    def num_channels(self) -> int:
        """Number of layer/head channels represented by the reference."""

        return int(self.calibration_channel_score.shape[1])

    @property
    def channel_values(self) -> np.ndarray:
        """Concise alias for the persisted per-channel reference matrix."""

        return self.calibration_channel_score

    @property
    def fusion_reference(self) -> np.ndarray:
        """Concise alias for the persisted scalar fusion reference."""

        return self.calibration_fusion_stat

    @property
    def sorted_channel_score(self) -> np.ndarray:
        """Float32 channel ECDF order cached for runtime scoring."""

        return self._sorted_channel_score

    @property
    def sorted_fusion_stat(self) -> np.ndarray:
        """Float32 scalar ECDF order cached for runtime scoring."""

        return self._sorted_fusion_stat


@dataclass(frozen=True, slots=True)
class CalibratedScores:
    """Primary scores and their auditable intermediate quantities."""

    score: np.ndarray
    fusion_stat: np.ndarray
    global_p_value: np.ndarray
    channel_p_value: np.ndarray


def leave_one_out_upper_tail_p(calibration_values) -> np.ndarray:
    """Compute conservative per-channel leave-one-out upper-tail ``p``-values.

    For calibration row ``i`` and channel ``c`` the returned value is

    ``(1 + #{j != i: x[j,c] >= x[i,c]}) / K``.

    The greater-than-or-equal comparison treats ties conservatively.  Including
    the pseudocount makes the expression equal to the inclusive upper-tail rank
    divided by ``K`` and guarantees a non-zero result.
    """

    values = _float_matrix(
        calibration_values,
        name="calibration_values",
        minimum_rows=2,
    )
    row_count, channel_count = values.shape
    probability = np.empty_like(values, dtype=np.float32)
    for channel in range(channel_count):
        ordered = np.sort(values[:, channel])
        first_tie = np.searchsorted(
            ordered,
            values[:, channel],
            side="left",
        )
        inclusive_tail_count = row_count - first_tie
        probability[:, channel] = inclusive_tail_count.astype(np.float32) / np.float32(
            row_count
        )
    return probability


def upper_tail_p(calibration_values, values) -> np.ndarray:
    """Compare test rows with a frozen per-channel empirical upper tail.

    The finite-sample value is ``(1 + #{calibration >= test}) / (K + 1)``.
    Consequently an observation beyond all calibration rows receives the
    smallest attainable, but never zero, probability.
    """

    calibration = _float_matrix(
        calibration_values,
        name="calibration_values",
        minimum_rows=2,
    )
    tested = _float_matrix(
        values,
        name="values",
        minimum_rows=1,
        columns=calibration.shape[1],
    )
    ordered = np.sort(calibration, axis=0)
    return _upper_tail_p_from_sorted(ordered, tested)


def _upper_tail_p_from_sorted(ordered, values) -> np.ndarray:
    """Apply the channel ECDF using a pre-sorted float32 reference."""

    # ``CalibrationReference`` validates and owns this immutable matrix once.
    # Do not scan all K*C reference entries again for every scored response.
    calibration = np.asarray(ordered, dtype=np.float32)
    if (
        calibration.ndim != 2
        or len(calibration) < 2
        or calibration.shape[1] < 1
    ):
        raise ValueError("sorted calibration values have invalid geometry")
    tested = _float_matrix(
        values,
        name="values",
        minimum_rows=1,
        columns=calibration.shape[1],
    )
    row_count, channel_count = calibration.shape
    probability = np.empty_like(tested, dtype=np.float32)
    denominator = np.float32(row_count + 1)
    for channel in range(channel_count):
        first_tie = np.searchsorted(
            calibration[:, channel],
            tested[:, channel],
            side="left",
        )
        tail_count = row_count - first_tie
        probability[:, channel] = (
            tail_count.astype(np.float32) + np.float32(1.0)
        ) / denominator
    return probability


def cauchy_combination_statistic(channel_p_value) -> np.ndarray:
    """Combine dependent channel ``p``-values without fitting channel weights.

    Small channel probabilities map to a large positive statistic.  Inputs are
    clipped only at ``float32`` machine precision so exact values of one do not
    produce negative infinity through the tangent transform.
    """

    probability = _float_matrix(
        channel_p_value,
        name="channel_p_value",
        minimum_rows=1,
    )
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError("channel_p_value must lie in [0, 1]")
    epsilon = np.float32(np.finfo(np.float32).eps)
    clipped = np.clip(probability, epsilon, np.float32(1.0) - epsilon)
    angle = np.float32(np.pi) * (np.float32(0.5) - clipped)
    transformed = np.tan(angle).astype(np.float32, copy=False)
    statistic = np.mean(transformed, axis=1, dtype=np.float32)
    if not bool(np.isfinite(statistic).all()):
        raise ValueError("Cauchy combination produced a non-finite statistic")
    return statistic


# A concise public spelling retained alongside the explicit statistical name.
cauchy_combination = cauchy_combination_statistic


def _global_upper_tail_p(reference, values) -> np.ndarray:
    """Finite-sample upper-tail probabilities for scalar fusion statistics."""

    calibration = _float_vector(
        reference,
        name="fusion_reference",
        minimum_rows=2,
    )
    tested = _float_vector(values, name="fusion_stat", minimum_rows=1)
    return _global_upper_tail_p_from_sorted(np.sort(calibration), tested)


def _global_upper_tail_p_from_sorted(reference, values) -> np.ndarray:
    """Apply the scalar ECDF using one pre-sorted float32 reference."""

    ordered = np.asarray(reference, dtype=np.float32)
    if ordered.ndim != 1 or len(ordered) < 2:
        raise ValueError("sorted fusion reference has invalid geometry")
    tested = _float_vector(values, name="fusion_stat", minimum_rows=1)
    first_tie = np.searchsorted(ordered, tested, side="left")
    tail_count = len(ordered) - first_tie
    return (
        tail_count.astype(np.float32) + np.float32(1.0)
    ) / np.float32(len(ordered) + 1)


def build_calibration(
    channel_values,
    fusion_values=None,
    *,
    storage_dtype=np.float16,
) -> CalibrationReference:
    """Build a label-free two-stage calibration reference.

    Parameters
    ----------
    channel_values:
        The first, per-channel calibration stream with shape ``[K, C]``.
    fusion_values:
        Optional independent stream with shape ``[K2, C]``.  These rows are
        first compared with ``channel_values`` and then fused.  If omitted, the
        channel stream is fused from leave-one-out probabilities.
    storage_dtype:
        Floating dtype for the channel matrix.  ``float16`` is the default to
        keep a 1024-channel reference compact; values are promoted to
        ``float32`` before every ECDF or fusion operation.
    """

    channel = _float_matrix(
        channel_values,
        name="channel_values",
        minimum_rows=2,
    )
    dtype = np.dtype(storage_dtype)
    if dtype.kind != "f" or dtype.itemsize not in {2, 4, 8}:
        raise ValueError("storage_dtype must be a floating dtype")
    stored_channel = channel.astype(dtype)
    if not bool(np.isfinite(stored_channel).all()):
        raise ValueError("channel_values overflow in the requested storage dtype")

    # Derive all reference statistics from the stored values.  A save/load
    # round trip therefore cannot change the fitted calibration distribution.
    computed_channel = stored_channel.astype(np.float32)
    if fusion_values is None:
        fusion_probability = leave_one_out_upper_tail_p(computed_channel)
        independent = False
    else:
        independent_values = _float_matrix(
            fusion_values,
            name="fusion_values",
            minimum_rows=2,
            columns=computed_channel.shape[1],
        )
        fusion_probability = upper_tail_p(computed_channel, independent_values)
        independent = True
    fusion_stat = cauchy_combination_statistic(fusion_probability)
    return CalibrationReference(
        calibration_channel_score=stored_channel,
        calibration_fusion_stat=fusion_stat,
        independent_fusion_reference=independent,
    )


def score_channels(reference: CalibrationReference, channel_values) -> CalibratedScores:
    """Calibrate and fuse test channel scores without using any labels.

    ``channel_values`` must have shape ``[N, C]``.  The primary score is exactly
    ``-log(global_p_value)``; the other arrays are returned so score artifacts
    can retain a complete audit trail without exposing alternative primary
    detectors.
    """

    if not isinstance(reference, CalibrationReference):
        raise TypeError("reference must be a CalibrationReference")
    values = _float_matrix(
        channel_values,
        name="channel_values",
        minimum_rows=1,
        columns=reference.num_channels,
    )
    channel_probability = _upper_tail_p_from_sorted(
        reference.sorted_channel_score,
        values,
    )
    fusion_stat = cauchy_combination_statistic(channel_probability)
    global_probability = _global_upper_tail_p_from_sorted(
        reference.sorted_fusion_stat,
        fusion_stat,
    )
    score = -np.log(global_probability).astype(np.float32, copy=False)
    if not bool(np.isfinite(score).all()):
        raise ValueError("calibration produced a non-finite primary score")
    return CalibratedScores(
        score=score,
        fusion_stat=fusion_stat,
        global_p_value=global_probability,
        channel_p_value=channel_probability,
    )


__all__ = [
    "AlignedFloat32Reservoir",
    "CalibratedScores",
    "CalibrationReference",
    "build_calibration",
    "cauchy_combination",
    "cauchy_combination_statistic",
    "leave_one_out_upper_tail_p",
    "score_channels",
    "upper_tail_p",
]
