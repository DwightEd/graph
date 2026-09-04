"""Token-level hallucination onset matching used only during evaluation."""

from __future__ import annotations

import numpy as np


def positive_onsets(label) -> np.ndarray:
    label = np.asarray(label, dtype=bool)
    return np.flatnonzero(label & ~np.r_[False, label[:-1]])


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return 1.0
    median = np.median(finite)
    scale = 1.4826 * np.median(np.abs(finite - median))
    if scale <= 1e-12:
        scale = np.std(finite)
    return float(scale) if scale > 1e-12 else 1.0


def match_onsets(
    label,
    token,
    *,
    window: int = 5,
    caliper: int = 32,
    covariates: dict[str, np.ndarray] | None = None,
    boundary: np.ndarray | None = None,
):
    """Match every hallucination onset to a clean token in the same response.

    Token identity remains the first preference.  Optional pre-outcome
    covariates then balance uncertainty and boundary status without entering
    capture or event discovery.
    """

    label = np.asarray(label, dtype=bool)
    token = np.asarray(token)
    if token.shape != label.shape:
        raise ValueError("token and label must have the same shape")
    boundary = (
        np.zeros_like(label)
        if boundary is None
        else np.asarray(boundary, dtype=bool)
    )
    if boundary.shape != label.shape:
        raise ValueError("boundary and label must have the same shape")
    covariates = {} if covariates is None else {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in covariates.items()
    }
    for name, value in covariates.items():
        if value.shape != label.shape:
            raise ValueError(f"covariate {name!r} must have the same shape as label")
    scales = {name: _robust_scale(value) for name, value in covariates.items()}

    candidates = [
        index
        for index in range(window, len(label) - window)
        if not label[index - window : index + window + 1].any()
    ]
    used: set[int] = set()
    pairs = []
    for onset in positive_onsets(label):
        if onset < window or onset + window >= len(label):
            continue
        available = [
            index
            for index in candidates
            if index not in used and abs(index - onset) <= caliper
        ]
        if not available:
            continue

        def distance(index: int) -> tuple[float, ...]:
            covariate_distance = 0.0
            for name, values in covariates.items():
                left, right = values[onset], values[index]
                if np.isfinite(left) and np.isfinite(right):
                    covariate_distance += abs(left - right) / scales[name]
            return (
                float(token[index] != token[onset]),
                float(boundary[index] != boundary[onset]),
                covariate_distance,
                abs(index - onset) / max(caliper, 1),
                float(index),
            )

        control = min(available, key=distance)
        used.add(control)
        pairs.append((int(onset), int(control)))
    return pairs


def match_events(
    event,
    token,
    *,
    window: int = 2,
    caliper: int = 32,
    covariates: dict[str, np.ndarray] | None = None,
    boundary: np.ndarray | None = None,
):
    """Match label-free internal events to non-events in the same response.

    Controls cannot fall within ``window`` positions of any selected event.
    Matching uses only coordinates and baseline quantities available before
    opening hallucination labels.
    """

    event = np.asarray(event, dtype=bool)
    token = np.asarray(token)
    if token.shape != event.shape:
        raise ValueError("token and event must have the same shape")
    boundary = (
        np.zeros_like(event)
        if boundary is None
        else np.asarray(boundary, dtype=bool)
    )
    if boundary.shape != event.shape:
        raise ValueError("boundary and event must have the same shape")
    covariates = {} if covariates is None else {
        str(name): np.asarray(value, dtype=np.float64)
        for name, value in covariates.items()
    }
    for name, value in covariates.items():
        if value.shape != event.shape:
            raise ValueError(f"covariate {name!r} must have the same shape as event")
    scales = {name: _robust_scale(value) for name, value in covariates.items()}

    candidates = [
        index
        for index in range(window, len(event) - window)
        if not event[index - window : index + window + 1].any()
    ]
    used: set[int] = set()
    pairs = []
    for selected in np.flatnonzero(event):
        if selected < window or selected + window >= len(event):
            continue
        available = [
            index
            for index in candidates
            if index not in used and abs(index - selected) <= caliper
        ]
        if not available:
            continue

        def distance(index: int) -> tuple[float, ...]:
            covariate_distance = 0.0
            for name, values in covariates.items():
                left, right = values[selected], values[index]
                if np.isfinite(left) and np.isfinite(right):
                    covariate_distance += abs(left - right) / scales[name]
            return (
                float(token[index] != token[selected]),
                float(boundary[index] != boundary[selected]),
                covariate_distance,
                abs(index - selected) / max(caliper, 1),
                float(index),
            )

        control = min(available, key=distance)
        used.add(control)
        pairs.append((int(selected), int(control)))
    return pairs
