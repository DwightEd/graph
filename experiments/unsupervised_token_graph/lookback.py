"""Label-free local-to-remote lookback and information diagnostics.

The input is the compact reanchor response-query attention tensor with shape
``[layers, heads, response_queries, total_tokens]``. No hallucination labels,
future response tokens, or answer-region assumptions are used here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LookbackResult:
    waad: np.ndarray
    fai: np.ndarray
    local_mass: np.ndarray
    remote_history_mass: np.ndarray
    prompt_mass: np.ndarray
    evidence_entropy: np.ndarray
    evidence_concentration: np.ndarray
    distribution_shift: np.ndarray
    lookback_ratio: np.ndarray
    valid_history: np.ndarray
    event: np.ndarray
    event_strength: np.ndarray


class LookbackAnalyzer:
    """Compute reanchor events from local-to-remote attention transitions."""

    def __init__(self, window=10, horizon_low=10, horizon_high=100, event_quantile=.9, cooldown=2,
                 layers=None, heads=None):
        self.window = window
        self.horizon_low = horizon_low
        self.horizon_high = horizon_high
        self.event_quantile = event_quantile
        self.cooldown = cooldown
        self.layers = layers
        self.heads = heads

    def run(self, attention, prompt_length):
        attention = np.asarray(attention, dtype=np.float64)
        layers, heads, response_queries, total_tokens = attention.shape
        layer_ids = np.arange(layers) if self.layers is None else np.asarray(self.layers, dtype=int)
        head_ids = np.arange(heads) if self.heads is None else np.asarray(self.heads, dtype=int)
        attention = attention[np.ix_(layer_ids, head_ids, np.arange(response_queries), np.arange(total_tokens))]
        p = int(prompt_length)
        response_tokens = response_queries
        mean = attention.mean(axis=1)
        row_index = p - 1 + np.arange(response_tokens)
        rows = mean[:, np.arange(response_queries), :]
        rows = rows.mean(axis=0)
        prompt_mass = rows[:, :p].sum(-1)
        local = np.zeros(response_tokens)
        remote = np.zeros(response_tokens)
        waad = np.zeros(response_tokens)
        evidence_entropy = np.zeros(response_tokens)
        evidence_concentration = np.zeros(response_tokens)
        for t, q in enumerate(row_index):
            response_keys = np.arange(p, min(q + 1, total_tokens))
            if len(response_keys):
                response_indices = response_keys - p
                distances = q - response_keys
                local_mask = distances <= self.window
                local[t] = rows[t, response_keys[local_mask]].sum()
                remote[t] = rows[t, response_keys[~local_mask]].sum()
                waad[t] = np.sum(rows[t, response_keys] * np.minimum(distances, self.window))
            evidence = rows[t, :p]
            mass = evidence.sum()
            if mass:
                prob = evidence / mass
                evidence_entropy[t] = -np.sum(prob * np.log2(np.maximum(prob, 1e-12)))
                evidence_concentration[t] = np.sum(prob * prob)
        shift = np.zeros(response_tokens)
        for t in range(1, response_tokens):
            previous = rows[t - 1, :p]
            current = rows[t, :p]
            previous /= previous.sum() or 1.
            current /= current.sum() or 1.
            midpoint = .5 * (previous + current)
            shift[t] = .5 * np.sum(previous * np.log2(np.maximum(previous, 1e-12) / np.maximum(midpoint, 1e-12)))
            shift[t] += .5 * np.sum(current * np.log2(np.maximum(current, 1e-12) / np.maximum(midpoint, 1e-12)))
        fai = np.zeros(response_tokens)
        for source in range(response_tokens):
            lo = source + self.horizon_low
            hi = min(response_tokens, source + self.horizon_high + 1)
            if lo < hi:
                key = p + source
                fai[source] = rows[lo:hi, key].mean()
        history = local + remote
        ratio = np.divide(prompt_mass + remote, prompt_mass + history,
                  out=np.zeros_like(prompt_mass), where=(prompt_mass + history) > 0)
        valid_history = np.arange(response_tokens) >= 2
        delta = np.r_[0., np.diff(ratio)]
        candidates = ratio[valid_history]
        threshold = np.quantile(candidates, self.event_quantile) if len(candidates) else np.inf
        event = valid_history & (ratio >= threshold) & (delta > 0)
        for t in range(response_tokens):
            if event[t] and t:
                event[t + 1:min(response_tokens, t + self.cooldown + 1)] = False
        strength = np.maximum(delta, 0.) * np.log1p(ratio)
        return LookbackResult(waad, fai, local, remote, prompt_mass, evidence_entropy,
                              evidence_concentration, shift, ratio, valid_history, event, strength)


class InformationDiagnostics:
    """Label-join diagnostics for frozen structural scores."""

    @staticmethod
    def histogram_kl(left, right, bins=20):
        values = np.concatenate((np.asarray(left), np.asarray(right)))
        edges = np.quantile(values, np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return {"kl": 0., "js": 0., "overlap": 1., "bins": 1}
        p, _ = np.histogram(left, bins=edges); q, _ = np.histogram(right, bins=edges)
        p = (p + 1e-8) / (p.sum() + 1e-8 * len(p)); q = (q + 1e-8) / (q.sum() + 1e-8 * len(q))
        midpoint = .5 * (p + q)
        kl = np.sum(p * np.log2(p / q))
        js = .5 * np.sum(p * np.log2(p / midpoint)) + .5 * np.sum(q * np.log2(q / midpoint))
        return {"kl": float(kl), "js": float(js), "overlap": float(np.minimum(p, q).sum()), "bins": len(p)}

    @staticmethod
    def empirical_logloss(scores, labels):
        scores = np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
        labels = np.asarray(labels, dtype=float)
        return float(np.mean(-labels * np.log(scores) - (1 - labels) * np.log1p(-scores)))