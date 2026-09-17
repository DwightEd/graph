"""Layer/head-resolved ROUTING observations on fixed normal/error pairs.

No detector or LLM forward. Attention is thresholded; prompt is not verified
factual evidence. Gold intervals are diagnostic coordinates, not detector input.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import load_graph, load_predictions, save_scores, write_json


METRICS = ('self_mass', 'prompt_mass', 'history_mass', 'within_mass',
           'within_excess', 'noncopy_excess', 'history_hhi', 'shared_history_js', 'retained_mass')
PHASES = ('pre', 'onset', 'early', 'middle', 'late', 'post', 'onset_minus_pre')


def finite_average(values, axis=0):
    count = np.isfinite(values).sum(axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def query_index(graph):
    source, target = graph['edge_index']
    if np.any(source < 0) or np.any(source >= target) or np.any(target < int(graph['prompt_length'])) or np.any(target >= len(graph['x'])):
        raise ValueError('Expected saved past-to-response routing edges')
    if graph['x'].shape[1] != int(graph['layers']) * int(graph['heads']):
        raise ValueError('Flattened channel dimension does not equal layers times heads')
    order = np.argsort(target, kind='stable')
    return order, target[order]


def query_edges(graph, index, query):
    order, target = index
    left, right = np.searchsorted(target, [query, query + 1])
    edges = order[left:right]
    return graph['edge_index'][0, edges], graph['edge_attr'][edges].astype(float)


def history_js(graph, index, query):
    """Consecutive query distributions on the SAME already-existing history keys.

Exclude the newly available key q-1. Missing retained mass => unobserved, not zero.
"""
    prompt = int(graph['prompt_length'])
    channels = graph['x'].shape[1]
    if query <= prompt + 1:
        return np.full(channels, np.nan)
    source, values = query_edges(graph, index, query)
    old_source, old_values = query_edges(graph, index, query - 1)
    keep = (source >= prompt) & (source < query - 1)
    old_keep = (old_source >= prompt) & (old_source < query - 1)
    universe = np.union1d(source[keep], old_source[old_keep])
    current = np.zeros((len(universe), channels))
    previous = np.zeros_like(current)
    current[np.searchsorted(universe, source[keep])] = values[keep]
    previous[np.searchsorted(universe, old_source[old_keep])] = old_values[old_keep]
    a, b = current.sum(axis=0), previous.sum(axis=0)
    current /= np.maximum(a, 1e-30)
    previous /= np.maximum(b, 1e-30)
    middle = (current + previous) / 2
    divergence = .5 * (current * np.log(np.maximum(current, 1e-30) / np.maximum(middle, 1e-30))).sum(axis=0)
    divergence += .5 * (previous * np.log(np.maximum(previous, 1e-30) / np.maximum(middle, 1e-30))).sum(axis=0)
    divergence[(a <= 0) | (b <= 0)] = np.nan
    return divergence


def conditional_excess(source, weights, query, start, end, prompt, token_ids):
    """Exact endpoint-randomization expectation at fixed lag band + token-ID copy.

Includes all eligible past response keys, including zero/unsaved slots. Lag 1
is its own group. A pure previous-token chain therefore has zero excess.
"""
    candidates = np.arange(prompt, query)
    if not len(candidates):
        return np.zeros(weights.shape[1]), np.zeros(weights.shape[1])
    bands = np.ceil(np.log2(query - candidates)).astype(int)
    copies = token_ids[candidates] == token_ids[query]
    groups = 2 * bands + copies.astype(int)
    size = int(groups.max()) + 1
    all_count = np.bincount(groups, minlength=size)
    inside = (candidates >= start) & (candidates < end)
    inside_count = np.bincount(groups[inside], minlength=size)
    fraction = np.divide(inside_count, all_count, out=np.zeros(size), where=all_count > 0)
    selected = source >= prompt
    group_weights = np.zeros((size, weights.shape[1]))
    np.add.at(group_weights, groups[source[selected] - prompt], weights[selected])
    actual = weights[(source >= start) & (source < end)].sum(axis=0)
    noncopy = token_ids[source] != token_ids[query]
    actual_noncopy = weights[(source >= start) & (source < end) & noncopy].sum(axis=0)
    expected = (group_weights * fraction[:, None]).sum(axis=0)
    expected_noncopy = (group_weights[::2] * fraction[::2, None]).sum(axis=0)
    return actual - expected, actual_noncopy - expected_noncopy


def measure_query(graph, index, sample, token, start, end):
    prompt = int(graph['prompt_length'])
    query, left, right = prompt + token, prompt + start, prompt + end
    source, weights = query_edges(graph, index, query)
    history = weights[source >= prompt]
    history_mass = history.sum(axis=0)
    concentration = np.divide(np.square(history).sum(axis=0), history_mass ** 2,
        out=np.full(weights.shape[1], np.nan), where=history_mass > 0)
    excess, noncopy = conditional_excess(source, weights, query, left, right, prompt, sample['token_ids'])
    diagonal = graph['x'][query].astype(float)
    values = (diagonal, weights[source < prompt].sum(axis=0), history_mass,
        weights[(source >= left) & (source < right)].sum(axis=0), excess, noncopy,
        concentration, history_js(graph, index, query), weights.sum(axis=0) + diagonal)
    return np.stack(values).reshape(len(METRICS), int(graph['layers']), int(graph['heads']))


def phase_offsets(pair, labels, window):
    """Use common observable normal positions before/after BOTH intervals."""
    error, normal, length = pair['error_start'], pair['normal_start'], pair['length']
    pre = [offset for offset in range(-window, 0)
           if min(error, normal) + offset >= 0 and not labels[error + offset] and not labels[normal + offset]]
    post = []
    for offset in range(length, length + window):
        if max(error, normal) + offset >= len(labels) or labels[error + offset] or labels[normal + offset]:
            break
        post.append(offset)
    offsets = np.arange(length)
    thirds = np.minimum(2, ((offsets + .5) * 3 / length).astype(int))
    return (pre, [0], offsets[thirds == 0], offsets[thirds == 1], offsets[thirds == 2], post)


def measure_pair(graph, index, sample, pair, window):
    shape = (2, len(PHASES), len(METRICS), int(graph['layers']), int(graph['heads']))
    values = np.full(shape, np.nan, np.float32)
    observed = np.zeros(shape, np.int32)
    phases = phase_offsets(pair, sample['gold'], window)
    caches = [{}, {}]
    for phase, offsets in enumerate(phases):
        if not len(offsets):
            continue
        for role, start in enumerate((pair['error_start'], pair['normal_start'])):
            for offset in offsets:
                if int(offset) not in caches[role]:
                    caches[role][int(offset)] = measure_query(
                        graph, index, sample, start + int(offset), start, start + pair['length'])
        stack = np.stack([[cache[int(offset)] for offset in offsets] for cache in caches])
        common = np.isfinite(stack[0]) & np.isfinite(stack[1])
        stack = np.where(common[None], stack, np.nan)
        values[:, phase] = finite_average(stack, axis=1)
        observed[:, phase] = common.sum(axis=0)
    values[:, -1] = values[:, 1] - values[:, 0]
    observed[:, -1] = np.minimum(observed[:, 0], observed[:, 1])
    return values, observed


def summarize_routes(entries, output, bootstrap):
    """Paired differences first, then equal-source means; heads aren't replicates."""
    values = np.stack([row['values'] for row in entries]).astype(float)
    sources = np.asarray([row['source_id'] for row in entries])
    common = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
    values = np.where(common[:, None], values, np.nan)
    delta = values[:, 0] - values[:, 1]
    per_source = np.stack([finite_average(delta[sources == source]) for source in np.unique(sources)])
    mean = finite_average(per_source)
    error_mean, normal_mean = finite_average(values[:, 0]), finite_average(values[:, 1])
    rows = []
    for phase, phase_name in enumerate(PHASES):
        for metric, metric_name in enumerate(METRICS):
            block = per_source[:, phase, metric]
            rng = np.random.default_rng(42)
            draws = [finite_average(block[rng.integers(len(block), size=len(block))]) for _ in range(bootstrap if len(block) > 1 else 0)]
            low, high = percentile_bounds(draws, block.shape[1:])
            low[np.isfinite(block).sum(axis=0) < 2] = np.nan
            high[np.isfinite(block).sum(axis=0) < 2] = np.nan
            for layer, head in np.ndindex(block.shape[1:]):
                rows.append(dict(phase=phase_name, metric=metric_name, llm_layer=layer, llm_head=head,
                    pairs=int(common[:, phase, metric, layer, head].sum()),
                    sources=int(np.isfinite(block[:, layer, head]).sum()),
                    error_mean=error_mean[phase, metric, layer, head],
                    normal_mean=normal_mean[phase, metric, layer, head],
                    source_mean_delta=mean[phase, metric, layer, head], low=low[layer, head], high=high[layer, head]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'layer_head_routes.csv.gz', index=False)
    depth_contrasts(entries, output, bootstrap)
    return frame


def percentile_bounds(draws, shape):
    low, high = np.full(shape, np.nan), np.full(shape, np.nan)
    if draws:
        stack = np.stack(draws)
        valid = np.isfinite(stack).any(axis=0)
        low[valid], high[valid] = np.nanquantile(stack[:, valid], [.025, .975], axis=0)
    return low, high


def depth_contrasts(entries, output, bootstrap):
    """Two independent axes: token phase and LLM depth. Not GNN update depth."""
    from .evaluate import source_interval

    rows = []
    for entry in entries:
        delta = entry['values'][0] - entry['values'][1]
        layer_groups = np.array_split(np.arange(delta.shape[2]), 3)
        for depth, layers in zip(('shallow', 'middle', 'deep'), layer_groups):
            if not len(layers):
                continue
            for name, phase, metric in (('onset_prompt_change', 6, 1), ('late_noncopy_excess', 4, 5),
                                         ('late_history_concentration', 4, 6), ('late_shared_key_js', 4, 7)):
                value = finite_average(delta[phase, metric, layers].reshape(-1))
                rows.append(dict(id=entry['id'], source_id=entry['source_id'], error_start=entry['error_start'],
                    contrast=name, depth=depth, first_layer=int(layers[0]), last_layer=int(layers[-1]), value=value))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'depth_contrasts.csv', index=False)
    summary = []
    for (name, depth), group in frame.groupby(['contrast', 'depth']):
        measured = source_interval(group.dropna(subset=['value']), 'value', bootstrap)
        summary.append(dict(contrast=name, depth=depth, **measured))
    pd.DataFrame(summary).to_csv(output / 'depth_summary.csv', index=False)


def run_routes(args, output, pairs):
    from .visualize import route_plots

    pairs = [pair for pair in pairs if pair['tier'] == args.pair_tier]
    if not pairs:
        raise ValueError('Requested tier has no matched pairs; no substitute controls are selected')
    predictions = load_predictions(Path(args.root) / 'charm_in' / 'test')
    grouped = {}
    for pair in pairs:
        grouped.setdefault(str(pair['id']), []).append(pair)
    write_json(output / 'pairs.json', pairs)
    entries = []
    for record in tqdm(predictions, desc='layer/head paired routes', unit='answer'):
        if str(record['id']) not in grouped:
            continue
        graph, sample = load_graph(record, args.prepared)
        if any(not np.array_equal(record[k], sample[k]) for k in ('gold', 'offsets', 'spans')):
            raise ValueError('Route graph and original prediction coordinates differ')
        index = query_index(graph)
        for pair in grouped[str(record['id'])]:
            if str(pair['source_id']) != str(record['source_id']):
                raise ValueError('Pair source differs from graph source')
            a, b, length = pair['error_start'], pair['normal_start'], pair['length']
            if not sample['gold'][a:a+length].all() or sample['gold'][b:b+length].any():
                raise ValueError('Pair does not match graph labels')
            values, counts = measure_pair(graph, index, sample, pair, args.window)
            path = output / 'pairs' / f'{record["id"]}_{a}.npz'
            save_scores(path, means=values, observed=counts, metrics=np.asarray(METRICS), phases=np.asarray(PHASES),
                id=np.asarray(str(record['id'])), source_id=np.asarray(str(record['source_id'])),
                error_start=np.asarray(a), normal_start=np.asarray(b), length=np.asarray(length))
            entries.append(dict(id=str(record['id']), source_id=str(record['source_id']), error_start=a, values=values))
    if len(entries) != len(pairs):
        raise ValueError('Some fixed pairs have no original predictions')
    frame = summarize_routes(entries, output, args.bootstrap)
    route_plots(frame, output / 'figures')
    write_json(output / 'protocol.json', dict(pairs=len(entries), window=args.window, metrics=METRICS, phases=PHASES,
        input='existing thresholded attention graph; no LLM/GNN forward',
        axes='original LLM layer/head x token phase, NOT GNN depth',
        uncertainty='exploratory source-bootstrap intervals, no multiple-comparison significance claim',
        limitations=['Prompt mass is not independently verified evidence support.',
            'No value vectors, residual readout, gradients, or native LLM intervention are observed.',
            'Same head index at different layers is not assumed to be the same functional head.',
            'Shared-key JS is missing without retained common history mass; never padded with zero.',
            'Attention missing below the graph threshold is not recovered. Retained mass is reported.',
            'Lock-in and early failure remain hypotheses, not automatic diagnoses.']))
