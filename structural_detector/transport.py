"""Head-resolved propagation on the emitted-token dependency DAG (not causal effects)."""
import numpy as np

MODES = ('real', 'uniform', 'permuted', 'one_hop', 'no_edges')


def validate(seed, edges):
    a, w = np.asarray(seed, float), np.asarray(edges, float)
    if a.ndim != 1 or w.ndim != 3 or len(a) != len(w) or not w.shape[1] or not w.shape[2]:
        raise ValueError('seed [T] and edges [T,channels,lag] required')
    if not np.isfinite(a).all() or not np.isfinite(w).all() or (a < 0).any() or (a > 1).any() or (w < 0).any():
        raise ValueError('finite probabilities and nonnegative edges required')
    if (w.sum(-1) > 1.00001).any():
        raise ValueError('local mass exceeds one; normalize full attention rows, never local rows')
    for t in range(min(len(a), w.shape[2])):
        if w[t, :, t:].any():
            raise ValueError('edge points before response start; lag d maps to token t-d')
    return a, w


def propagate(seed, edges, mode='real', random_seed=20260914):
    """U[t,k]=sum_d A[t,k,d]*R[t-d,k]; R=a+(1-a)*U.

    Missing mass escapes to non-local history/prompt, not renormalized. Each
    physical (layer,head) is an independent channel until the final readout.
    This recurrence is an absorbing-walk risk proxy, not native WV/WO/JVP flow.
    It permits arbitrary-length chains of local edges. No duration smoothing.
    """
    a, w = validate(seed, edges)
    if mode not in MODES:
        raise ValueError('unknown transport control')
    u, state = np.zeros(w.shape[:2]), np.zeros(w.shape[:2])
    parent = np.full(w.shape[:2], -1, dtype=np.int32)
    rng = np.random.default_rng(random_seed)
    for t in range(len(a)):
        n = min(t, w.shape[2])
        if n and mode != 'no_edges':
            weights = w[t, :, :n]
            if mode == 'uniform':
                weights = np.broadcast_to(weights.sum(-1, keepdims=True) / n, weights.shape)
            elif mode == 'permuted':
                # Same permutation across heads; preserve every head's mass and weights.
                weights = weights[:, rng.permutation(n)]
            indices = t - 1 - np.arange(n)
            previous = np.broadcast_to(a[indices], (w.shape[1], n)) if mode == 'one_hop' else state[indices].T
            messages = weights * previous
            u[t] = messages.sum(-1)
            chosen = messages.argmax(-1)
            parent[t] = np.where(u[t] > 0, indices[chosen], -1)
        state[t] = a[t] + (1 - a[t]) * np.minimum(u[t], 1.)
    return u, state, parent


def onset_features(values):
    """The entry detector ONLY sees current entropy and its past difference."""
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 5 or not len(x) or not np.isfinite(x).all():
        raise ValueError('five finite S10-compatible columns required')
    h = x[:, 0]
    return np.column_stack((h, np.r_[0., np.diff(h)]))


def continuation_features(seed, edges, evidence, mode='real'):
    """Inherited risk vs total local/evidence mass; no labels or gold state."""
    u, state, parent = propagate(seed, edges, mode)
    evidence = np.asarray(evidence, float)
    if evidence.shape != u.shape or not np.isfinite(evidence).all() or (evidence < 0).any() or (evidence > 1.00001).any():
        raise ValueError('evidence mass must align with physical channels')
    x = np.column_stack((u, edges.sum(-1), evidence))
    return x, u, state, parent


def mixture(onset, conditional_continuation):
    """P(error)=P(onset)+(1-P(onset))*P(continuation | not onset)."""
    a, c = np.asarray(onset), np.asarray(conditional_continuation)
    if a.shape != c.shape or not np.isfinite(a).all() or not np.isfinite(c).all() or (a < 0).any() or (a > 1).any() or (c < 0).any() or (c > 1).any():
        raise ValueError('aligned probabilities required')
    return a + (1 - a) * c
