"""Label-free temporal state inference for route observations.

The detector sees only ``(contraction, takeover)`` observations.  It fits one
sticky diagonal-Gaussian HMM per task and names the fitted states from their
route geometry, never from hallucination labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

STATE_NAMES = ("exploration", "grounded_focus", "captured")


def logsumexp(
    values: np.ndarray, axis: int | tuple[int, ...] | None = None
) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    summed = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    if axis is None:
        return summed.reshape(())
    return np.squeeze(summed, axis=axis)


def valid_runs(
    sequence: np.ndarray, valid_mask: np.ndarray | None = None
) -> list[tuple[slice, np.ndarray]]:
    """Return contiguous finite runs without bridging excluded early tokens."""

    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 2:
        raise ValueError("route observations must have shape [tokens, 2]")

    valid = np.isfinite(sequence).all(axis=1)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    changes = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return [
        (slice(start, stop), sequence[start:stop])
        for start, stop in changes.reshape(-1, 2)
    ]


class StickyRouteHMM:
    """Three-state unsupervised HMM for route contraction and takeover."""

    exploration_state = 0
    grounded_state = 1
    captured_state = 2
    state_names = STATE_NAMES

    def __init__(
        self,
        n_iter: int = 100,
        stickiness: float = 10.0,
        tolerance: float = 1e-5,
        variance_floor: float = 1e-4,
    ) -> None:
        self.n_iter = n_iter
        self.stickiness = stickiness
        self.tolerance = tolerance
        self.variance_floor = variance_floor

        self.initial_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        valid_masks: Sequence[np.ndarray] | None = None,
    ) -> StickyRouteHMM:
        """Fit from unlabeled sequences; invalid rows are excluded events."""

        if valid_masks is None:
            valid_masks = [None] * len(sequences)
        runs = [
            run
            for sequence, valid in zip(sequences, valid_masks, strict=True)
            for _, run in valid_runs(sequence, valid)
        ]
        if not runs:
            raise ValueError("at least one valid route observation is required")

        observations = np.concatenate(runs)
        if len(observations) < 3:
            raise ValueError("three route states require at least three observations")

        self.initial_, self.transition_, self.means_, self.variances_ = (
            self._initialize(observations)
        )

        previous_likelihood = -np.inf
        for _ in range(self.n_iter):
            initial_counts = np.zeros(3)
            transition_counts = np.zeros((3, 3))
            state_counts = np.zeros(3)
            state_sums = np.zeros((3, 2))
            state_squares = np.zeros((3, 2))
            likelihood = 0.0

            for run in runs:
                emission = self._log_emission(run)
                forward, run_likelihood = self._forward(emission)
                backward = self._backward(emission)
                posterior = np.exp(forward + backward - logsumexp(forward[-1], axis=0))

                initial_counts += posterior[0]
                state_counts += posterior.sum(axis=0)
                state_sums += posterior.T @ run
                state_squares += posterior.T @ np.square(run)
                likelihood += run_likelihood

                if len(run) > 1:
                    pair_log = (
                        forward[:-1, :, None]
                        + np.log(self.transition_)[None, :, :]
                        + emission[1:, None, :]
                        + backward[1:, None, :]
                    )
                    pair_log -= logsumexp(pair_log, axis=(1, 2))[:, None, None]
                    transition_counts += np.exp(pair_log).sum(axis=0)

            self.initial_ = (initial_counts + 1.0) / (initial_counts.sum() + 3.0)
            transition_prior = np.ones((3, 3)) + self.stickiness * np.eye(3)
            self.transition_ = transition_counts + transition_prior
            self.transition_ /= self.transition_.sum(axis=1, keepdims=True)

            occupied = state_counts > 1e-8
            self.means_[occupied] = state_sums[occupied] / state_counts[occupied, None]
            self.variances_[occupied] = state_squares[occupied] / state_counts[
                occupied, None
            ] - np.square(self.means_[occupied])
            self.variances_ = np.maximum(self.variances_, self.variance_floor)

            if (
                np.isfinite(previous_likelihood)
                and abs(likelihood - previous_likelihood) < self.tolerance
            ):
                break
            previous_likelihood = likelihood

        self._name_states()
        return self

    def filtered_posterior(
        self, sequence: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Return online posteriors; each row depends only on its prefix."""

        self._require_fit()
        sequence = np.asarray(sequence, dtype=np.float64)
        posterior = np.full((len(sequence), 3), np.nan)
        for location, run in valid_runs(sequence, valid_mask):
            forward, _ = self._forward(self._log_emission(run))
            forward -= logsumexp(forward, axis=1)[:, None]
            posterior[location] = np.exp(forward)
        return posterior

    def smoothed_posterior(
        self, sequence: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Return full-sequence posteriors for diagnostics and plots."""

        self._require_fit()
        sequence = np.asarray(sequence, dtype=np.float64)
        posterior = np.full((len(sequence), 3), np.nan)
        for location, run in valid_runs(sequence, valid_mask):
            emission = self._log_emission(run)
            forward, _ = self._forward(emission)
            backward = self._backward(emission)
            joint = forward + backward
            joint -= logsumexp(joint, axis=1)[:, None]
            posterior[location] = np.exp(joint)
        return posterior

    def score(
        self, sequence: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Primary online mechanism-risk score: filtered captured posterior."""

        return self.filtered_posterior(sequence, valid_mask)[:, self.captured_state]

    def independent_posterior(
        self, sequence: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Posterior from the same emissions with token order removed."""

        self._require_fit()
        sequence = np.asarray(sequence, dtype=np.float64)
        posterior = np.full((len(sequence), 3), np.nan)
        prior = self.stationary_probability()
        for location, run in valid_runs(sequence, valid_mask):
            value = self._log_emission(run) + np.log(prior)[None, :]
            value -= logsumexp(value, axis=1)[:, None]
            posterior[location] = np.exp(value)
        return posterior

    def independent_score(
        self, sequence: np.ndarray, valid_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Captured-state control without transition or persistence evidence."""

        return self.independent_posterior(sequence, valid_mask)[:, self.captured_state]

    def stationary_probability(self) -> np.ndarray:
        """Return the fitted chain's state occupancy used by the token control."""

        self._require_fit()
        probability = np.full(3, 1.0 / 3.0)
        for _ in range(1000):
            updated = probability @ self.transition_
            if np.max(np.abs(updated - probability)) < 1e-12:
                break
            probability = updated
        return probability / probability.sum()

    def expected_dwell_time(self) -> np.ndarray:
        """Expected consecutive tokens in each fitted state."""

        self._require_fit()
        diagonal = np.diag(self.transition_)
        return 1.0 / np.maximum(1.0 - diagonal, 1e-12)

    def save(self, path: str | Path) -> None:
        self._require_fit()
        np.savez(
            path,
            initial=self.initial_,
            transition=self.transition_,
            means=self.means_,
            variances=self.variances_,
            expected_dwell_time=self.expected_dwell_time(),
        )

    @classmethod
    def load(cls, path: str | Path) -> StickyRouteHMM:
        model = cls()
        with np.load(path) as parameters:
            model.initial_ = parameters["initial"]
            model.transition_ = parameters["transition"]
            model.means_ = parameters["means"]
            model.variances_ = parameters["variances"]
        return model

    def _initialize(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scale = np.maximum(observations.std(axis=0), np.sqrt(self.variance_floor))
        standardized = (observations - observations.mean(axis=0)) / scale

        exploration_count = max(1, len(observations) // 3)
        low_contraction = np.argsort(observations[:, 0])[:exploration_count]
        remaining = np.setdiff1d(np.arange(len(observations)), low_contraction)
        takeover_order = remaining[np.argsort(observations[remaining, 1])]
        centers = observations[
            [
                low_contraction[len(low_contraction) // 2],
                takeover_order[len(takeover_order) // 4],
                takeover_order[(3 * len(takeover_order)) // 4],
            ]
        ]

        assignments = np.full(len(observations), -1, dtype=np.int64)
        for _ in range(20):
            distances = np.square(
                standardized[:, None, :]
                - (centers - observations.mean(axis=0))[None, :, :] / scale
            ).sum(axis=2)
            updated = np.argmin(distances, axis=1)
            if np.array_equal(updated, assignments):
                break
            assignments = updated
            for state in range(3):
                if np.any(assignments == state):
                    centers[state] = observations[assignments == state].mean(axis=0)

        global_variance = np.maximum(observations.var(axis=0), self.variance_floor)
        variances = np.repeat(global_variance[None, :], 3, axis=0)
        for state in range(3):
            selected = observations[assignments == state]
            if len(selected) > 1:
                variances[state] = np.maximum(selected.var(axis=0), self.variance_floor)

        initial = np.full(3, 1.0 / 3.0)
        transition = np.ones((3, 3)) + self.stickiness * np.eye(3)
        transition /= transition.sum(axis=1, keepdims=True)
        return initial, transition, centers, variances

    def _log_emission(self, observations: np.ndarray) -> np.ndarray:
        difference = observations[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            2.0 * np.log(2.0 * np.pi)
            + np.log(self.variances_).sum(axis=1)[None, :]
            + (np.square(difference) / self.variances_[None, :, :]).sum(axis=2)
        )

    def _forward(self, emission: np.ndarray) -> tuple[np.ndarray, float]:
        forward = np.empty_like(emission)
        forward[0] = np.log(self.initial_) + emission[0]
        log_transition = np.log(self.transition_)
        for token in range(1, len(emission)):
            forward[token] = emission[token] + logsumexp(
                forward[token - 1, :, None] + log_transition, axis=0
            )
        likelihood = float(logsumexp(forward[-1], axis=0))
        return forward, likelihood

    def _backward(self, emission: np.ndarray) -> np.ndarray:
        backward = np.zeros_like(emission)
        log_transition = np.log(self.transition_)
        for token in range(len(emission) - 2, -1, -1):
            backward[token] = logsumexp(
                log_transition
                + emission[token + 1][None, :]
                + backward[token + 1][None, :],
                axis=1,
            )
        return backward

    def _name_states(self) -> None:
        exploration = int(np.argmin(self.means_[:, 0]))
        remaining = [state for state in range(3) if state != exploration]
        captured = max(remaining, key=lambda state: self.means_[state, 1])
        grounded = next(state for state in remaining if state != captured)
        order = np.array([exploration, grounded, captured])

        self.initial_ = self.initial_[order]
        self.transition_ = self.transition_[order][:, order]
        self.means_ = self.means_[order]
        self.variances_ = self.variances_[order]

    def _require_fit(self) -> None:
        if self.means_ is None:
            raise RuntimeError("fit the route-state model before scoring")
