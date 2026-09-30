"""CPU readout for source-qualified token evidence, without truth-label inputs.

The native measurements and automatic relation proposals are separate inputs;
this module does not infer evidence from attention or reconstruct node features.
Matrices use [earlier carrier q, later target t], with zero-based token addresses.
"""

import networkx as nx
import numpy as np


TOKEN_COST = .5
MIN_REFERENCE_SOURCES = 32
REFERENCE_TAIL = .05


def graph_energy(labels, unary, directed, continuity):
    """Evaluate the original energy, including signed unary terms."""
    labels = np.asarray(labels, dtype=float)
    inherited = labels[:, None] * (1 - labels[None, :])
    return float(unary @ labels + np.sum(directed * inherited)
                 + continuity[1:] @ np.abs(np.diff(labels)))


def _energy_arrays(source, support, carrier, directed, continuity, cost):
    arrays = [np.asarray(value, dtype=float) for value in
              (source, support, carrier, directed, continuity)]
    source, support, carrier, directed, continuity = arrays
    count = source.size
    if any(value.shape != (count,) for value in arrays[:3] + arrays[4:]):
        raise ValueError('source, support, carrier and continuity must have shape [T]')
    if directed.shape != (count, count):
        raise ValueError('directed must have shape [T, T]')
    if any(not np.isfinite(value).all() or np.any(value < 0) for value in arrays):
        raise ValueError('evidence and pair capacities must be finite and nonnegative')
    if np.any(np.tril(directed)) or (count and continuity[0] != 0):
        raise ValueError('directed edges require q < t; continuity[0] must be zero')
    if not np.isfinite(cost) or cost <= 0:
        raise ValueError('token cost must be finite and positive')
    return cost + support - source - carrier, directed, continuity


def _cut_graph(unary, directed, continuity):
    count = len(unary)
    source, sink = count, count + 1
    graph = nx.DiGraph()
    graph.add_nodes_from(range(count + 2))
    for token, value in enumerate(unary):
        graph.add_edge(source, token, capacity=max(-float(value), 0.))
        graph.add_edge(token, sink, capacity=max(float(value), 0.))
    for sender, receiver in zip(*np.nonzero(directed)):
        graph.add_edge(int(sender), int(receiver), capacity=float(directed[sender, receiver]))
    for token in range(1, count):
        value = float(continuity[token])
        previous = graph.get_edge_data(token - 1, token, {}).get('capacity', 0.)
        graph.add_edge(token - 1, token, capacity=previous + value)
        graph.add_edge(token, token - 1, capacity=value)
    return graph


def _cut_labels(graph, count, forced=None):
    source, sink = count, count + 1
    if forced is not None:
        token, label, capacity = forced
        graph = graph.copy()
        edge = (source, token) if label else (token, sink)
        graph.edges[edge]['capacity'] += capacity
    residual = nx.algorithms.flow.preflow_push(graph, source, sink)
    # Source-reachable residual nodes give the inclusion-minimal source set
    # among all minimum cuts. No epsilon perturbs neutral-node decisions.
    reachable, pending = {source}, [source]
    while pending:
        node = pending.pop()
        for neighbour, edge in residual[node].items():
            if neighbour not in reachable and edge['capacity'] > edge['flow']:
                reachable.add(neighbour)
                pending.append(neighbour)
    return np.array([token in reachable for token in range(count)], dtype=bool)


def solve_energy(source, support, carrier, directed, continuity, cost=TOKEN_COST):
    """Solve E(z) and every exact min-marginal using at most T+1 cuts.

    E = (cost + support - source - carrier) @ z
        + sum(w[q,t] z[q](1-z[t])) + sum(b[t] |z[t]-z[t-1]|).
    Positive min-marginal favours membership. Opposite constrained solutions
    are retained for auditing; their energies exclude the forcing capacity.
    """
    unary, directed, continuity = _energy_arrays(
        source, support, carrier, directed, continuity, cost)
    count = len(unary)
    graph = _cut_graph(unary, directed, continuity)
    labels = _cut_labels(graph, count)
    energy = graph_energy(labels, unary, directed, continuity)
    forcing = 1 + np.abs(unary).sum() + directed.sum() + continuity.sum()
    opposite = np.empty((count, count), dtype=bool)
    opposite_energy = np.empty(count)
    for token in range(count):
        opposite[token] = _cut_labels(graph, count, (token, not labels[token], forcing))
        opposite_energy[token] = graph_energy(opposite[token], unary, directed, continuity)
    margin = np.where(labels, opposite_energy - energy, energy - opposite_energy)
    return dict(labels=labels, energy=energy, min_marginal=margin,
                opposite_labels=opposite, opposite_energy=opposite_energy,
                unary=unary, cut_count=count + 1)


def _reference_groups(records):
    """Deduplicate (source_id, record_id); long sources retain total weight one."""
    unique, groups = {}, {}
    for record in records:
        key = (record['source_id'], record['record_id'])
        value = float(record['value'])
        if not np.isfinite(value):
            raise ValueError('reference values must be finite')
        if key in unique and unique[key] != value:
            raise ValueError(f'conflicting duplicate reference record: {key}')
        unique[key] = value
    for (source, _), value in unique.items():
        groups.setdefault(source, []).append(value)
    return groups, len(unique)


def reference_scale(raw, records, noise_floor=1e-8):
    """Source-weighted empirical tail scale; not a p-value or FPR guarantee.

    The caller supplies one frozen matched family/pool, with target sources
    excluded. Unresolved pools return NaN, never apparent zero-risk scores.
    """
    raw = np.asarray(raw, dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(noise_floor) or noise_floor < 1e-8:
        raise ValueError('raw scores must be finite and noise_floor must be >= 1e-8')
    groups, count = _reference_groups(records)
    resolved = len(groups) >= MIN_REFERENCE_SOURCES
    tail = np.full(raw.shape, np.nan)
    score = np.full(raw.shape, np.nan)
    if resolved:
        exceed = np.zeros(raw.shape)
        for values in groups.values():
            exceed += np.mean(np.asarray(values).reshape((-1,) + (1,) * raw.ndim)
                              >= raw, axis=0)
        tail = (1 + exceed) / (len(groups) + 1)
        score = np.maximum(np.log(REFERENCE_TAIL / tail), 0.)
        score = np.where(raw <= noise_floor, 0., score)
    return dict(score=score, tail=tail, resolved=resolved,
                below_numerical_resolution=raw <= noise_floor,
                source_count=len(groups), record_count=count,
                duplicate_count=len(records) - count,
                minimum_tail=1 / (len(groups) + 1))


def reference_threshold(records, quantile=.975):
    """inf{x: F_source_equal(x) >= quantile}, clamped to nonnegative."""
    groups, _ = _reference_groups(records)
    if not groups or not 0 < quantile <= 1:
        raise ValueError('a nonempty independent reference and quantile in (0,1] are required')
    weighted = [(value, 1 / len(values)) for values in groups.values() for value in values]
    values, weights = np.asarray(sorted(weighted), dtype=float).T
    cumulative = np.cumsum(weights) / np.sum(weights)
    index = min(np.searchsorted(cumulative, quantile), len(values) - 1)
    return max(0., float(values[index]))


def _event_arrays(event, count):
    vectors = ('edit_mask', 'support', 'carrier', 'continuity', 'scope', 'aligned',
               'reference_resolved', 'measurement_complete')
    result = {name: np.asarray(event[name]) for name in vectors}
    if any(value.shape != (count,) for value in result.values()):
        raise ValueError('all event token fields must have shape [T]')
    for name in ('edit_mask', 'aligned', 'reference_resolved', 'measurement_complete'):
        if not np.isin(result[name], [0, 1]).all():
            raise ValueError(f'{name} must contain booleans or exact 0/1, not unknown sentinels')
        result[name] = result[name].astype(bool)
    if not np.isin(result['scope'], [-1, 0, 1]).all():
        raise ValueError('scope must contain only unknown=-1, excluded=0, expressed=1')
    for name in ('external_conflict', 'external_support', 'source_reference_resolved'):
        if not isinstance(event[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be a boolean')
    if isinstance(event['anchor'], (bool, np.bool_)) or not isinstance(event['anchor'], (int, np.integer)):
        raise ValueError('anchor must be an integer token address, not a boolean or float')
    if not 0 <= event['anchor'] < count:
        raise ValueError('anchor must be an original answer token address')
    edited = np.flatnonzero(result['edit_mask'])
    if not len(edited) or event['anchor'] != edited[0]:
        raise ValueError('anchor must be the earliest edited original token or insertion boundary')
    result['directed'] = np.asarray(event['directed'], dtype=float)
    if result['directed'].shape != (count, count):
        raise ValueError('event directed field must have shape [T, T]')
    for name in ('source_conflict', 'source_support'):
        if not np.isfinite(event[name]) or event[name] < 0:
            raise ValueError('event source strengths must be finite and nonnegative')
    return result


def _event_capacities(event, values, cost):
    source_ready = bool(event['source_reference_resolved'])
    conflict = source_ready and event['external_conflict'] and not event['external_support']
    support = source_ready and event['external_support'] and not event['external_conflict']
    strength = float(event['source_conflict']) if conflict else 0.
    direct = strength - (float(event['source_support']) if support else 0.) - cost
    after_fork = np.arange(len(values['scope'])) >= event['anchor']
    usable = (after_fork & (values['scope'] == 1) & values['aligned'].astype(bool)
              & values['reference_resolved'].astype(bool)
              & values['measurement_complete'].astype(bool) & (strength > 0))
    source = np.zeros(len(usable))
    source[event['anchor']] = strength
    carrier = np.where(usable, values['carrier'], 0.).astype(float)
    carrier[-1] = 0.  # The final token has no future target to influence.
    directed = np.where(usable[:, None] & usable[None, :], values['directed'], 0.)
    outgoing = directed.sum(axis=1)
    scale = np.minimum(1., np.divide(2 * cost, outgoing,
                                   out=np.ones_like(outgoing), where=outgoing > 0))
    directed = directed * scale[:, None]
    continuity = np.minimum(values['continuity'], .75 * cost).astype(float)
    pair_usable = usable[1:] & usable[:-1]
    pair_supported = np.maximum(values['support'][1:], values['support'][:-1]) > 0
    continuity[1:] = np.where(pair_usable & ~pair_supported, continuity[1:], 0.)
    continuity[0] = 0.
    return source, carrier, directed, continuity, direct, usable


def token_runs(mask):
    """Return half-open token intervals for one event, without window averaging."""
    transitions = np.diff(np.pad(np.asarray(mask, dtype=int), (1, 1)))
    return list(zip(np.flatnonzero(transitions == 1).tolist(),
                    np.flatnonzero(transitions == -1).tolist()))


def score_event(event, token_count, direct_threshold=0., graph_threshold=0., cost=TOKEN_COST):
    """Apply eligibility and capacity bounds, retaining each event's own spans."""
    values = _event_arrays(event, token_count)
    source, carrier, directed, continuity, direct, usable = _event_capacities(event, values, cost)
    solved = solve_energy(source, values['support'], carrier, directed, continuity, cost)
    edit = values['edit_mask'].astype(bool)
    qualified = event['source_reference_resolved'] and (
        bool(event['external_conflict']) != bool(event['external_support']))
    alarm = (edit & (direct > direct_threshold)) | (solved['min_marginal'] > graph_threshold)
    masks = {name: values[name].astype(bool) for name in
             ('aligned', 'reference_resolved', 'measurement_complete')}
    masks.update(scope=values['scope'] == 1, scope_unknown=values['scope'] == -1)
    return dict(event_id=event['event_id'], **solved, direct=direct, edit_mask=edit,
                usable=usable, evidence=usable | (edit & qualified), masks=masks,
                source_reference_resolved=bool(event['source_reference_resolved']),
                capacities=dict(source=source, support=values['support'], carrier=carrier,
                                directed=directed, continuity=continuity),
                spans=token_runs(alarm))


def score_events(events, token_count, direct_threshold=0., graph_threshold=0., cost=TOKEN_COST):
    """Combine frozen, qualified evidence without deleting independent direct alarms.

    Every event supplies source strengths, external qualification flags, anchor,
    edit_mask[T], independently externally qualified support[T], transformed
    carrier/w/b, and scope/alignment/reference/measurement masks. No raw model
    state, candidate generation or gold boundary is inferred here. Incomplete
    measurements produce explicitly provisional arrays and complete=False.
    """
    if min(direct_threshold, graph_threshold) < 0 or not np.isfinite(
            [direct_threshold, graph_threshold]).all():
        raise ValueError('independent channel thresholds must be finite and nonnegative')
    direct = np.full(token_count, -np.inf)
    graph = np.full(token_count, -np.inf)
    evidence = np.zeros(token_count, dtype=bool)
    measured = np.ones(token_count, dtype=bool)
    event_outputs = []
    for event in events:
        solved = score_event(event, token_count, direct_threshold, graph_threshold, cost)
        edit = solved['edit_mask']
        direct[edit] = np.maximum(direct[edit], solved['direct'])
        graph = np.maximum(graph, solved['min_marginal'])
        evidence |= solved['evidence']
        measured &= solved['masks']['measurement_complete']
        event_outputs.append(solved)
    if not events:
        graph.fill(-cost)
    direct[~np.isfinite(direct)] = -cost
    risk = np.maximum(direct - direct_threshold, graph - graph_threshold)
    return dict(direct=direct, graph=graph, risk=risk, alarm=risk > 0,
                low_evidence=~evidence, measurement_complete=measured,
                complete=bool(measured.all()), events=event_outputs)
