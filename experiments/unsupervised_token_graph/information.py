"""Information identities on a token-routing surrogate, not semantic MI.

Z is a prompt root or an explicit self/unknown terminal. C is the last-hop
carrier. Layers and heads are never averaged before this computation.
"""

import numpy as np
from scipy.special import entr, rel_entr


LOG2 = np.log(2.)


def entropy_bits(probability, axis=-1):
    return entr(np.asarray(probability)).sum(axis=axis) / LOG2


def js_bits(left, right, alpha=.5):
    """Weighted JS of two probability vectors; empty conditionals are undefined."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.sum() == 0 or right.sum() == 0:
        return np.nan
    left, right = left / left.sum(), right / right.sum()
    mixture = alpha * left + (1 - alpha) * right
    return float((alpha * rel_entr(left, mixture).sum()
                  + (1 - alpha) * rel_entr(right, mixture).sum()) / LOG2)


def row_js(left_keys, left_values, right_keys, right_values):
    """Compare fixed absolute key identities, including unobserved mass."""
    keys = np.union1d(left_keys, right_keys)
    left, right = np.zeros(len(keys) + 1), np.zeros(len(keys) + 1)
    left[np.searchsorted(keys, left_keys)] = left_values
    right[np.searchsorted(keys, right_keys)] = right_values
    left[-1], right[-1] = max(0., 1 - left.sum()), max(0., 1 - right.sum())
    return js_bits(left, right)


def future_influence(channel, horizon_low=10, horizon_high=100):
    """OFFLINE ONLY: attention received by the token predicted at each row.

    Distances are measured between predicted token positions, not CSR rows.
    No observation window is NaN, not an observed zero.
    """
    positions = channel.prediction_positions
    result = np.full(len(positions), np.nan)
    count = np.zeros(len(positions), dtype=np.int64)
    for t, source in enumerate(positions):
        chosen = (positions - source >= horizon_low) & (positions - source <= horizon_high)
        chosen &= channel.queries >= source
        if source < channel.attention.shape[1] and chosen.any():
            count[t] = int(chosen.sum())
            result[t] = float(channel.attention[chosen, source].sum()) / count[t]
    return result, count


class SourceFlow:
    """Absorb a backward walk into prompt roots, self boundaries, or unknown.

    This is a same-channel token-DAG model. Recursion across token positions
    must not be described as tracing the true inter-layer WV/WO computation.
    """

    def __init__(self, root_bins=None, seed=20260915):
        self.root_bins = root_bins
        self.seed = seed

    def run(self, channel, control="real", keep_roots=False):
        p, queries = channel.prompt_length, channel.queries
        roots = p if self.root_bins is None else min(p, self.root_bins)
        if roots < 1 or control not in ("real", "no_ancestry", "history_permuted"):
            raise ValueError("positive root count and a declared graph control are required")
        size, self_root, unknown_root = roots + 2, roots, roots + 1
        states = np.zeros((len(queries), size))
        path_numerator = np.zeros(len(queries))
        state_entropy = np.zeros(len(queries))
        row_of = {int(q): i for i, q in enumerate(queries)}
        rng = np.random.default_rng(self.seed + 10007 * channel.layer + channel.head)
        names = ("direct_prompt_mass", "relay_prompt_mass", "prompt_reach", "self_terminal_mass",
                 "unknown_mass", "terminal_entropy_bits", "carrier_conditional_entropy_bits",
                 "carrier_information_bits", "prompt_root_entropy_bits", "prompt_path_length",
                 "source_mismatch_bits")
        result = {name: np.full(len(queries), np.nan) for name in names}
        for t, q in enumerate(queries):
            keys, weights = channel.row(t)
            direct = np.zeros(size)
            mask = keys < p
            np.add.at(direct, keys[mask] * roots // p, weights[mask])
            history = (keys >= p) & (keys < q)
            carriers, carrier_weights = keys[history].copy(), weights[history]
            if control == "history_permuted":
                # Preserve direct/self/missing mass and history weight multiset
                # within lag bands; do not claim to preserve exact distances.
                bands = np.searchsorted([1, 4, 16, 64], q - carriers, side="left")
                for band in np.unique(bands):
                    index = np.flatnonzero(bands == band)
                    carriers[index] = rng.permutation(carriers[index])
            relay = np.zeros(size)
            conditional_entropy = 0.
            path_numerator[t] = direct.sum()
            for carrier, weight in zip(carriers, carrier_weights):
                parent = row_of.get(int(carrier))
                if control == "no_ancestry":
                    origin = np.zeros(size); origin[self_root] = 1.
                elif parent is None:
                    origin = np.zeros(size); origin[unknown_root] = 1.
                else:
                    origin = states[parent]
                    path_numerator[t] += weight * (path_numerator[parent] + origin[:roots].sum())
                relay += weight * origin
                conditional_entropy += weight * state_entropy[parent] if parent is not None and control != "no_ancestry" else 0.
            state = direct + relay
            if q >= p:
                state[self_root] += weights[keys == q].sum()
            state[unknown_root] += max(0., 1. - weights.sum())
            states[t] = state
            direct_mass, relay_mass = direct[:roots].sum(), relay[:roots].sum()
            reach = direct_mass + relay_mass
            marginal_entropy = entropy_bits(state)
            state_entropy[t] = marginal_entropy
            result["direct_prompt_mass"][t] = direct_mass
            result["relay_prompt_mass"][t] = relay_mass
            result["prompt_reach"][t] = reach
            result["self_terminal_mass"][t] = state[self_root]
            result["unknown_mass"][t] = state[unknown_root]
            result["terminal_entropy_bits"][t] = marginal_entropy
            result["carrier_conditional_entropy_bits"][t] = conditional_entropy
            result["carrier_information_bits"][t] = max(0., marginal_entropy - conditional_entropy)
            if reach > 0:
                result["prompt_root_entropy_bits"][t] = entropy_bits(state[:roots] / reach)
                result["prompt_path_length"][t] = path_numerator[t] / reach
            if direct_mass > 0 and relay_mass > 0:
                result["source_mismatch_bits"][t] = js_bits(direct[:roots], relay[:roots], direct_mass / reach)
        if keep_roots:
            result["root_distribution"] = states
        return result
