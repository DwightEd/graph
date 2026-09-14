"""Two stages: label-free entropy seeds, then a substochastic local-reading graph.

No labels, sklearn, trained heads, gold boundaries, or generated future tokens.
Every physical (layer, head) is scored separately; token risk uses a fixed quantile.
This is a diagnostic graph over generation positions, NOT a Transformer JVP.
"""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np


@dataclass(frozen=True)
class Config:
    window: int = 16
    seed_tail: float = .05
    survival: float = .90
    channel_quantile: float = .90
    seed: int = 20260914
    min_reference: int = 128
    per_source: int = 512

    def __post_init__(self):
        if self.window < 1 or not 0 < self.seed_tail < 1 or not 0 < self.survival < 1:
            raise ValueError('positive window, 0<seed_tail<1 and 0<survival<1 required')
        if not 0 <= self.channel_quantile <= 1 or min(self.min_reference, self.per_source) < 1:
            raise ValueError('invalid channel quantile or reference budget')


def source_splits(rows, seed=20260914):
    """Official test remains test; split official train sources 80/20 without labels."""
    train = {str(r['source_id']) for r in rows if r['official_split'] == 'train'}
    test = {str(r['source_id']) for r in rows if r['official_split'] == 'test'}
    if train & test or any(r['official_split'] not in ('train', 'test') for r in rows):
        raise ValueError('invalid or overlapping official sources')
    ordered = sorted(train, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).digest())
    n = int(.8 * len(ordered))
    if n < 1 or n == len(ordered):
        raise ValueError('at least two train sources needed for reference/calibration')
    reference = set(ordered[:n])
    return {str(r['id']): ('test' if str(r['source_id']) in test else
                          'reference' if str(r['source_id']) in reference else 'calibration') for r in rows}


def group_key(row):
    return f"{row['task']}|{row['generator']}"


def position_bins(n):
    # Only current absolute position, not answer length, affects bin identity.
    return np.searchsorted([1, 4, 16, 64], np.arange(n), side='right')


def fit_reference(rows, read_entropy, config):
    """Source-balanced empirical survival distribution of UNLABELED reference tokens.

    Mixed correct/error reference data are allowed: tail means unusual, not false.
    A position-specific bin falls back only to the SAME task/generator reference.
    """
    pools = {}
    for row in rows:
        h = np.asarray(read_entropy(row), float)
        if h.ndim != 1 or not len(h) or not np.isfinite(h).all():
            raise ValueError('finite nonempty entropy reference required')
        bins, group, sid = position_bins(len(h)), group_key(row), str(row['source_id'])
        for b in [-1, *np.unique(bins).tolist()]:
            values = h if b == -1 else h[bins == b]
            pools.setdefault(f'{group}|{b}', {}).setdefault(sid, []).append(values)
    tables = {}
    for key, by_source in pools.items():
        chunks, weights = [], []
        for source in sorted(by_source):
            values = np.concatenate(by_source[source])
            if len(values) > config.per_source:
                values = values[np.linspace(0, len(values) - 1, config.per_source, dtype=int)]
            chunks.append(values)
            weights.append(np.full(len(values), 1. / len(values)))
        x, w = np.concatenate(chunks), np.concatenate(weights)
        order = np.argsort(x, kind='stable')
        w = w[order] / w.sum()
        tables[key] = dict(values=x[order].tolist(), cumulative=np.cumsum(w).tolist(),
                           count=len(x), sources=len(by_source))
    if not tables:
        raise ValueError('no unlabeled reference data')
    return dict(schema='entropy-reference-v1', config=asdict(config), tables=tables,
                reference_ids=[str(r['id']) for r in rows], labels_used=False)


def seed_scores(entropy, row, reference):
    h = np.asarray(entropy, float)
    if h.ndim != 1 or not np.isfinite(h).all():
        raise ValueError('finite entropy required')
    c = Config(**reference['config']); bins = position_bins(len(h))
    tail = np.empty(len(h)); used = np.empty(len(h), int)
    for b in np.unique(bins):
        key = f'{group_key(row)}|{b}'
        table = reference['tables'].get(key)
        if table is None or table['count'] < c.min_reference:
            key = f'{group_key(row)}|-1'; table = reference['tables'].get(key)
        if table is None:
            raise ValueError('no reference for task/generator ' + group_key(row))
        x, cumulative = np.asarray(table['values']), np.asarray(table['cumulative'])
        # Inclusive >= tie handling: a constant reference must not make all tokens seeds.
        index = np.searchsorted(x, h[bins == b], side='left')
        below = np.r_[0., cumulative][index]
        n = table['count']
        tail[bins == b] = (1 + n * np.maximum(0., 1 - below)) / (n + 1)
        used[bins == b] = int(key.rsplit('|', 1)[1])
    seed = np.maximum(0., 1 - tail / c.seed_tail)
    return dict(onset_rank=1 - tail, seed=seed, tail=tail, reference_bin=used)


def validate_edges(edges, n, window):
    a = np.asarray(edges, dtype=np.float64)
    if a.ndim != 4 or a.shape[0] != n or a.shape[-1] != window:
        raise ValueError('local_attention must have shape [T,layer,head,lag]')
    if not np.isfinite(a).all() or (a < 0).any() or (a.sum(-1) > 1.00001).any():
        raise ValueError('raw nonnegative, substochastic local rows required; no local normalization')
    for t in range(min(n, window)):
        if np.any(a[t, :, :, t:] != 0):
            raise ValueError('edge to current/future/unavailable response token')
    # Correct only <=1e-5 float-rounding overflow; rows with mass <1 are unchanged.
    a = a / np.maximum(1., a.sum(-1, keepdims=True))
    return a.reshape(n, -1, window)


def controlled_edges(a, mode, seed):
    out = a.copy(); n, _, width = a.shape
    rng = np.random.default_rng(seed)
    for t in range(1, n):
        k = min(t, width)
        if mode == 'uniform':
            out[t, :, :k] = a[t, :, :k].sum(-1, keepdims=True) / k
        elif mode == 'permuted':
            # Preserve lag1 and coarse lag groups, local mass and head identity.
            # The same permutation across heads also preserves their cross-head agreement.
            for lo, hi in ((1, 4), (4, 8), (8, width)):
                hi = min(hi, k)
                if hi - lo > 1:
                    p = rng.permutation(np.arange(lo, hi))
                    out[t, :, lo:hi] = a[t][:, p]
        elif mode != 'real':
            raise ValueError(mode)
    return out


def propagate(seeds, edges, config, *, recursive=True, detail=False):
    """u[t,c]=gamma*sum_lag A[t,c,lag]*r[t-lag]; r[t]=max(seed[t],quantile_c(u)).

    No noisy-OR or cumulative maximum: without a new seed risk cannot grow.
    Nonlocal/source/unmarked-history mass is NOT redistributed to marked tokens.
    Dominant parent/root is an explanation of the largest summand, not all paths.
    """
    seeds = np.asarray(seeds, float)
    if seeds.ndim != 1 or not np.isfinite(seeds).all() or (seeds < 0).any() or (seeds > 1).any():
        raise ValueError('seed scores must be finite in [0,1]')
    a = validate_edges(edges, len(seeds), config.window)
    n, channels, width = a.shape
    r = np.zeros((n, channels)); inherited = np.zeros_like(r)
    token_risk = np.zeros(n); token_origin = np.full(n, -1, np.int32)
    selected = np.zeros(n, np.int32)
    rank = int(np.ceil(config.channel_quantile * (channels - 1)))
    parent = np.full((n, channels), -1, np.int32) if detail else None
    origin = np.full((n, channels), -1, np.int32) if detail else None
    for t in range(n):
        k = min(t, width)
        if k:
            js = t - np.arange(1, k + 1)
            values = np.broadcast_to(token_risk[js] if recursive else seeds[js], (channels, k))
            terms = config.survival * a[t, :, :k] * values
            inherited[t] = terms.sum(-1)
            if detail:
                j = js[np.argmax(terms, axis=1)]
                active = inherited[t] > seeds[t]
                parent[t, active] = j[active]
                origin[t, active] = token_origin[j[active]] if recursive else j[active]
        r[t] = np.maximum(seeds[t], inherited[t])
        if detail:
            new = (seeds[t] > 0) & (seeds[t] >= inherited[t])
            origin[t, new] = t
        selected[t] = np.argsort(r[t], kind='stable')[rank]
        token_risk[t] = r[t, selected[t]]
        if detail:
            token_origin[t] = origin[t, selected[t]]
    # Order statistic: no averaging of heads before propagation or interpolation here.
    score = token_risk
    result = dict(risk=score, continuation=np.partition(inherited, rank, axis=1)[:, rank])
    if detail:
        result.update(channel_risk=r.astype(np.float32), channel_inherited=inherited.astype(np.float32),
                      dominant_parent=parent, dominant_origin=origin, selected_channel=selected.astype(np.int32))
    return result


def score_trace(trace, row, reference):
    config = Config(**reference['config'])
    entropy, edge = np.asarray(trace['entropy']), np.asarray(trace['local_attention'])
    seed = seed_scores(entropy, row, reference)
    primary = propagate(seed['seed'], edge, config, detail=True)
    a = validate_edges(edge, len(entropy), config.window)
    rseed = int.from_bytes(hashlib.sha256(f"{config.seed}:{row['id']}".encode()).digest()[:4], 'little')
    shape = edge.shape
    uniform = propagate(seed['seed'], controlled_edges(a, 'uniform', rseed).reshape(shape), config)
    permuted = propagate(seed['seed'], controlled_edges(a, 'permuted', rseed).reshape(shape), config)
    direct = propagate(seed['seed'], edge, config, recursive=False)
    scores = dict(onset_rank=seed['onset_rank'], seed_only=seed['seed'], reuse=primary['risk'],
                  continuation=primary['continuation'], single_hop=direct['risk'],
                  mass_matched_uniform=uniform['risk'], lag_group_permuted=permuted['risk'],
                  raw_entropy=entropy.copy(), raw_negative_margin=np.asarray(trace['negative_margin']).copy(),
                  raw_total_history_minus_source=(trace['history_mass'] - trace['source_mass']).mean(axis=(1, 2)),
                  raw_remote_history_minus_source=(trace['remote_mass'] - trace['source_mass']).mean(axis=(1, 2)),
                  raw_position=np.log1p(np.arange(len(entropy))))
    return scores, {**seed, **{k: v for k, v in primary.items() if k not in ('risk', 'continuation')}}
