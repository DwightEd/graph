"""Token-wise evidence contrasts; no neighbouring-token risk aggregation."""

import numpy as np

from experiments.native_support.unified.calibration import transform


METHODS = ('token_pair', 'odds_pair', 'odds_full')
PRIMARY = 'odds_full'
PROBABILITY_RESOLUTION = np.finfo(np.float32).eps


def log_odds(logp):
    """Regularize the complementary mass at the cached FP32 resolution.

    Cached log probabilities can round to zero. This is a resolution-limited
    contrast, not exact log odds when the complementary mass is unresolved.
    """
    logp = np.minimum(np.asarray(logp, dtype=float), 0.)
    complement = np.maximum(-np.expm1(logp), PROBABILITY_RESOLUTION)
    return logp - np.log(complement)


def contrasts(present_full, absent_full, present_local, absent_local, route):
    full = log_odds(absent_full) - log_odds(present_full)
    local = log_odds(absent_local) - log_odds(present_local)
    return dict(token_pair=.5 * (absent_full - present_full + absent_local - present_local),
                odds_pair=.5 * (full + local), odds_full=full, route=route)


def pack_contrasts(pack, metadata):
    observed = dict(zip(metadata['observations'], pack['observations'].T))
    context = dict(zip(metadata['context'], pack['context'].T))
    present_full = observed['with_full_logp']
    present_local = observed['with_local_logp']
    # Recover the original scalar measurements from the lossless pack layout.
    absent_full = present_full + observed['full_deviation'] + context['full_unit']
    absent_local = present_local + observed['local_deviation'] + context['local_unit']
    return contrasts(present_full, absent_full, present_local, absent_local, observed['route'])


def score_contrasts(values, scales):
    # Preserve the historical 3:1 source/route weight, without its window or
    # unit-mean broadcast. Ranks are reference percentiles, not error chances.
    route_rank = transform(values['route'], scales['route'])
    return {name: .75 * transform(values[name], scales[name]) + .25 * route_rank
            for name in METHODS}


def history_weights(root_effect, prompt_length):
    """Signed total root responses, not direct neural edges for path rollout."""
    count = len(root_effect)
    weights = np.zeros((count, count), dtype=float)
    weights[:, :count - 1] = root_effect[:, prompt_length:]
    return np.tril(weights, k=-1)


def matched_history(weights, token_ids, seed=None):
    """Preserve each receiver's signed weights within lag/repeated-ID strata.

    None returns the conditional expectation. Randomization preserves row
    strata and weight multisets, not each sender's outgoing degree.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros_like(weights)
    for target in range(1, len(weights)):
        parents = np.arange(target)
        lag = np.floor(np.log2(target - parents)).astype(int)
        repeated = token_ids[:target] == token_ids[target]
        strata = 2 * lag + repeated
        for group in np.unique(strata):
            indices = parents[strata == group]
            values = weights[target, indices]
            result[target, indices] = values.mean() if seed is None else rng.permutation(values)
    return result


def signed_context(weights, attributes, expected=False, token_ids=None, preserve_mass=False):
    """One-hop feature contexts; no recursive multiplication of total effects."""
    contexts = []
    for channel in (np.maximum(weights, 0), np.maximum(-weights, 0)):
        if expected:
            channel = matched_history(channel, token_ids)
        if not preserve_mass:
            mass = channel.sum(-1, keepdims=True)
            channel = np.divide(channel, mass, out=np.zeros_like(channel), where=mass > 0)
        contexts.append(channel @ attributes)
    return np.column_stack(contexts)


def attribution_lineage(root_effect, prompt_length):
    """CAGE-style positive lineage B + L C, not native causal message edges.

    Rows are output targets; prompt columns are original roots and L columns
    are strictly earlier answer tokens. Absolute total root derivatives are
    normalized once for this explanatory decomposition, never reinterpreted
    as local Jacobians or fed into the native graph detector.
    """
    roots = np.asarray(root_effect, dtype=float)
    count = len(roots)
    if roots.shape != (count, prompt_length + max(count - 1, 0)):
        raise ValueError('root_effect must have shape [T, prompt_length + T - 1]')
    if not np.isfinite(roots).all():
        raise ValueError('lineage requires completed finite root measurements')
    prompt = np.abs(roots[:, :prompt_length])
    history = np.abs(history_weights(roots, prompt_length))
    mass = prompt.sum(axis=1) + history.sum(axis=1)
    denominator = mass[:, None]
    prompt = np.divide(prompt, denominator, out=np.zeros_like(prompt), where=denominator > 0)
    history = np.divide(history, denominator, out=np.zeros_like(history), where=denominator > 0)
    lineage = prompt.copy()
    for target in range(count):
        lineage[target] += history[target, :target] @ lineage[:target]
    return dict(B=prompt, L=history, C=lineage, root_mass=mass, resolved=mass > 0)


def _native_statistic_inputs(eta, selection, aligned, valid):
    eta = np.asarray(eta, dtype=float)
    selection = np.asarray(selection)
    if not np.issubdtype(selection.dtype, np.integer) or np.any(selection < 0):
        raise ValueError('physical layer/head addresses must be nonnegative integers')
    if not np.isin(aligned, [0, 1]).all() or not np.isin(valid, [0, 1]).all():
        raise ValueError('alignment/measurement masks require exact booleans or 0/1')
    selection = selection.astype(int)
    aligned = np.asarray(aligned, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    count = len(aligned)
    if selection.ndim != 3 or selection.shape[0] != count or selection.shape[2] != 2:
        raise ValueError('selection must have shape [T, K, (layer, head)]')
    width = selection.shape[1]
    if eta.shape != (3, 3, count, width, count) or valid.shape != (count, width, count):
        raise ValueError('eta must have shape [3,3,T,K,T]; valid must have shape [T,K,T]')
    if width == 0 or any(len(set(map(tuple, row))) != width for row in selection):
        raise ValueError('each carrier requires a nonempty selection of distinct physical heads')
    causal = np.triu(np.ones((count, count), dtype=bool), 1)
    eligible = causal & aligned[:, None] & aligned[None, :]
    return eta, selection, aligned, valid, causal, eligible


def _directional_excess(eta, valid, causal, eligible):
    # Each P uses its own output direction. E/U are not projections under R.
    excess = np.full((3,) + eta.shape[2:], np.nan)
    complete = np.zeros_like(excess, dtype=bool)
    for direction in range(3):
        other = [donor for donor in range(3) if donor != direction]
        competitor = np.maximum(0., np.max(eta[direction, other], axis=0))
        current = np.maximum(eta[direction, direction] - competitor, 0.)
        measured = valid & np.isfinite(eta[direction]).all(axis=0)
        complete[direction] = ~eligible[:, None, :] | measured
        excess[direction] = np.where(eligible[:, None, :] & measured, current, np.nan)
        excess[direction] = np.where(causal[:, None, :], excess[direction], 0.)
    return excess, complete


def _pair_statistics(excess, selection, eligible):
    count = selection.shape[0]
    pair = np.zeros((3, count))
    complete = np.ones((3, count), dtype=bool)
    pair_eligible = np.zeros(count, dtype=bool)
    winner_target = np.full((3, count), -1, dtype=int)
    winner_head = np.full((3, count, 2), -1, dtype=int)
    for token in range(1, count):
        left = {tuple(head): index for index, head in enumerate(selection[token - 1])}
        right = {tuple(head): index for index, head in enumerate(selection[token])}
        heads = sorted(left.keys() & right.keys())
        targets = np.flatnonzero(eligible[token - 1] & eligible[token])
        if not heads or not len(targets):
            continue
        pair_eligible[token] = True
        candidates = np.stack([np.minimum(excess[:, token - 1, left[head]][:, targets],
                                          excess[:, token, right[head]][:, targets])
                               for head in heads], axis=-1)  # [P, future target, head]
        complete[:, token] = np.isfinite(candidates).all(axis=(1, 2))
        pair[:, token] = candidates.max(axis=(1, 2))
        for direction in np.flatnonzero(complete[:, token]):
            target_index, head_index = np.unravel_index(
                candidates[direction].argmax(), candidates.shape[1:])
            winner_target[direction, token] = targets[target_index]
            winner_head[direction, token] = heads[head_index]
    return pair, complete, pair_eligible, winner_target, winner_head


def selected_statistics(eta, selection, aligned, valid, donor_names=('R', 'E', 'U')):
    """Raw self-aligned R/E/U max/min statistics, before empirical scaling.

    eta[P, donor, q, selected_head, target] contains native signed effects.
    Outputs node[P,q], edge[P,q,t], pair[P,q] use physical-head identities.
    Missing eligible measurements remain NaN through the selected maximum;
    eligibility and completion are separate, so unaligned entries cannot be
    reported as measured zero effects. Noncausal entries are structural zeros.
    """
    if tuple(donor_names) != ('R', 'E', 'U'):
        raise ValueError('both eta direction/donor axes must be ordered R, E, U')
    eta, selection, aligned, valid, causal, eligible = _native_statistic_inputs(
        eta, selection, aligned, valid)
    excess, complete = _directional_excess(eta, valid, causal, eligible)
    edge = excess.max(axis=2)
    edge_complete = complete.all(axis=2)
    node = np.where(eligible[None], edge, 0.).max(axis=2)
    pair, pair_complete, pair_eligible, target, head = _pair_statistics(excess, selection, eligible)
    return dict(directions=('R', 'E', 'U'), excess=excess, node=node, edge=edge, pair=pair,
                edge_eligible=eligible, node_eligible=eligible.any(axis=1),
                pair_eligible=pair_eligible, aligned=aligned,
                structural_zero=~causal, edge_complete=edge_complete,
                node_complete=edge_complete.all(axis=2), pair_complete=pair_complete,
                pair_target=target, pair_head=head,
                complete=bool(complete.all()))
