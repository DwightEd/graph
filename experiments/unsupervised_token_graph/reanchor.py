"""Causal, per-head reanchor candidates on sparse absolute token coordinates."""

from dataclasses import dataclass

import numpy as np

from .information import entropy_bits, js_bits, row_js


@dataclass
class ReanchorResult:
    waad: np.ndarray
    local_mass: np.ndarray
    remote_history_mass: np.ndarray
    prompt_mass: np.ndarray
    prompt_entropy: np.ndarray
    prompt_concentration: np.ndarray
    distribution_shift: np.ndarray
    reanchor_ratio: np.ndarray
    valid_history: np.ndarray
    event: np.ndarray
    event_strength: np.ndarray
    threshold: np.ndarray
    prompt_gain: np.ndarray
    remote_gain: np.ndarray


class ReanchorAnalyzer:
    """A candidate event is a new routing change, not merely an aging edge.

    Thresholds use earlier strengths only. Distribution entropy is computed
    within each head, never on an averaged attention distribution.
    """

    def __init__(self, window=10, event_quantile=.9, cooldown=2, min_history=8):
        if window < 1 or cooldown < 0 or min_history < 1 or not 0 < event_quantile < 1:
            raise ValueError("invalid reanchor window or past-only event parameters")
        self.window = window
        self.event_quantile = event_quantile
        self.cooldown = cooldown
        self.min_history = min_history

    def run(self, channel):
        p, queries = channel.prompt_length, channel.queries
        n = len(queries)
        waad, local, remote, prompt = (np.zeros(n) for _ in range(4))
        source_entropy, concentration, shift = (np.full(n, np.nan) for _ in range(3))
        prompt_gain, remote_gain, strength = (np.zeros(n) for _ in range(3))
        valid = queries - p + 1 >= 2
        for t, q in enumerate(queries):
            keys, weights = channel.row(t)
            prompt_mask, history = keys < p, keys >= p
            local_mask = history & (keys > q - self.window)
            remote_mask = history & ~local_mask
            prompt[t] = weights[prompt_mask].sum()
            local[t], remote[t] = weights[local_mask].sum(), weights[remote_mask].sum()
            waad[t] = np.sum(weights[history] * np.minimum(q - keys[history], self.window))
            if prompt[t] > 0:
                distribution = weights[prompt_mask] / prompt[t]
                source_entropy[t] = entropy_bits(distribution)
                concentration[t] = (distribution ** 2).sum()
            if t and queries[t] == queries[t - 1] + 1:
                previous_keys, previous_weights = channel.row(t - 1)
                common = np.union1d(previous_keys[previous_keys < p], keys[prompt_mask])
                if prompt[t - 1] > 0 and prompt[t] > 0:
                    left, right = np.zeros(len(common)), np.zeros(len(common))
                    left[np.searchsorted(common, previous_keys[previous_keys < p])] = previous_weights[previous_keys < p]
                    right[np.searchsorted(common, keys[prompt_mask])] = weights[prompt_mask]
                    shift[t] = js_bits(left, right)
                prompt_gain[t] = prompt[t] - prompt[t - 1]
                # Hold the remote set at the PREVIOUS query's absolute boundary.
                boundary = queries[t - 1] - self.window
                old = (previous_keys >= p) & (previous_keys <= boundary)
                now = (keys >= p) & (keys <= boundary)
                remote_gain[t] = weights[now].sum() - previous_weights[old].sum()
                gain = max(0., prompt_gain[t]) + max(0., remote_gain[t])
                strength[t] = gain * row_js(previous_keys, previous_weights, keys, weights)
        observed = prompt + local + remote
        ratio = np.divide(prompt + remote, observed, out=np.full(n, np.nan), where=observed > 0)
        thresholds, event = np.full(n, np.nan), np.zeros(n, bool)
        last_event = -self.cooldown - 1
        for t in range(n):
            past = strength[:t][valid[:t]]
            if valid[t] and len(past) >= self.min_history:
                thresholds[t] = np.quantile(past, self.event_quantile)
                event[t] = strength[t] > thresholds[t] and t - last_event > self.cooldown
                if event[t]:
                    last_event = t
        return ReanchorResult(waad, local, remote, prompt, source_entropy, concentration, shift,
                              ratio, valid, event, strength, thresholds, prompt_gain, remote_gain)


class InformationDiagnostics:
    """Post-freeze distribution diagnostics; never used to construct a graph."""

    @staticmethod
    def histogram_kl(left, right, bins=20):
        from scipy.special import rel_entr
        left, right = np.asarray(left, float), np.asarray(right, float)
        if left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right):
            raise ValueError("two nonempty one-dimensional samples are required")
        values = np.concatenate((left, right))
        if not np.isfinite(values).all():
            raise ValueError("select valid observations before distribution diagnostics")
        unique = np.unique(values)
        if len(unique) == 1:
            return dict(kl=0., js=0., overlap=1., bins=1)
        if len(unique) <= bins:
            # Discrete values get their own bins, including an imbalanced 0/1 score.
            edges = np.r_[-np.inf, unique[:-1] + np.diff(unique) / 2, np.inf]
        else:
            interior = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
            edges = np.r_[-np.inf, interior, np.inf]
        a, _ = np.histogram(left, edges); b, _ = np.histogram(right, edges)
        a, b = a / a.sum(), b / b.sum()
        # KL may correctly be infinite when empirical supports do not overlap.
        return dict(kl=float(rel_entr(a, b).sum() / np.log(2.)), js=js_bits(a, b),
                    overlap=float(np.minimum(a, b).sum()), bins=len(a))

    @staticmethod
    def empirical_logloss(probabilities, labels):
        probabilities, labels = np.asarray(probabilities, float), np.asarray(labels)
        if probabilities.ndim != 1 or probabilities.shape != labels.shape:
            raise ValueError("one probability and one label per observation are required")
        if (not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))
                or not np.isin(labels, (0, 1)).all()):
            raise ValueError("logloss requires probabilities in [0,1] and binary labels")
        predicted = np.clip(probabilities, 1e-7, 1 - 1e-7)
        return float(np.mean(-labels * np.log(predicted) - (1 - labels) * np.log1p(-predicted)))
