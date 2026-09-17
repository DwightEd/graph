"""Explain node_only with its own self-attention vector, never graph neighbors."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .compare_models import aligned_models, comparison_tokens, matched_tokens, relative_regions
from .data import read_json, write_json, save_scores
from .evaluate import paired_scores, source_interval


def load_node_attributes(root, prepared, table):
    """Open x and alignment metadata ONLY; edge_attr and embeddings are not read."""
    blocks, geometry = [], None
    for identity, group in tqdm(table.groupby('id', sort=False), desc='read node attributes', unit='answer'):
        path = Path(prepared)/'graphs/test'/(str(identity)+'.npz')
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved['record_json']))
            prompt = int(saved['prompt_length'])
            current = int(saved['layers']), int(saved['heads'])
            attributes = saved['x'][prompt:].astype(np.float32)
            labels, offsets = saved['gold'], saved['offsets']
            text = str(saved['response'])
        if str(record['id']) != str(identity) or set(group.source_id) != {str(record['source_id'])}:
            raise ValueError('Node attribute identity mismatch')
        if not np.array_equal(group.token, np.arange(len(labels))) or not np.array_equal(group.gold, labels):
            raise ValueError('Node attributes and original scores are not token-aligned')
        if group.text.tolist() != [text[a:b] for a, b in offsets]:
            raise ValueError('Node attributes and original token text differ')
        if attributes.shape != (len(group), current[0]*current[1]) or not np.isfinite(attributes).all():
            raise ValueError('Invalid original layer/head node attributes')
        if geometry is not None and current != geometry:
            raise ValueError('Mixed observer geometries cannot share a channel axis')
        geometry = current
        blocks.append(attributes)
    return np.concatenate(blocks), geometry


def pair_indices(table, paired):
    index = pd.Series(np.arange(len(table)), index=pd.MultiIndex.from_frame(table[['id', 'token']]))
    error = [(str(r.id), int(r.token)) for r in paired.itertuples()]
    normal = [(str(r.id), int(r.normal_token)) for r in paired.itertuples()]
    return index.loc[error].to_numpy(), index.loc[normal].to_numpy()


def source_vectors(values, metadata):
    """Equal pair means inside each source, then equal-source inference."""
    pairs, sources = [], []
    for (source, _, _), indices in metadata.groupby(['source_id', 'id', 'pair_error_start']).indices.items():
        pairs.append(values[indices].mean(axis=0))
        sources.append(source)
    pairs, sources = np.asarray(pairs), np.asarray(sources)
    return np.stack([pairs[sources == source].mean(axis=0) for source in np.unique(sources)])


def vector_interval(values, bootstrap):
    mean = values.mean(axis=0)
    low, high = np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    if bootstrap and len(values) > 1:
        rng = np.random.default_rng(42)
        draws = [values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(bootstrap)]
        low, high = np.quantile(draws, [.025, .975], axis=0)
    return mean, low, high


def attribute_cohorts(paired):
    regions = relative_regions(paired.pair_offset, paired.pair_length)
    for region, mask in regions.items():
        yield region, 'all', mask
    back = regions['back_half']
    for name, mask in (('node_hit', paired.node_alarm), ('node_miss', ~paired.node_alarm),
                       ('full_only_hit', paired.full_alarm & ~paired.node_alarm),
                       ('node_only_hit', paired.node_alarm & ~paired.full_alarm)):
        yield 'back_half', name, back & np.asarray(mask)


def attribute_statistics(error, normal, paired, heads, bootstrap):
    rows, coverage = [], []
    for region, cohort, mask in attribute_cohorts(paired):
        meta = paired.loc[mask].reset_index(drop=True)
        coverage.append(dict(region=region, cohort=cohort, tokens=len(meta),
            pairs=len(meta[['id', 'pair_error_start']].drop_duplicates()), sources=meta.source_id.nunique()))
        if meta.empty:
            continue
        left, right = error[mask], normal[mask]
        deltas = source_vectors(left-right, meta)
        mean, low, high = vector_interval(deltas, bootstrap)
        wins = source_vectors((left > right) + .5*(left == right), meta).mean(axis=0)
        for channel in range(error.shape[1]):
            rows.append(dict(region=region, cohort=cohort, llm_layer=channel//heads, llm_head=channel%heads,
                tokens=len(meta), sources=len(deltas), source_delta=mean[channel], low=low[channel], high=high[channel],
                pooled_error_mean=left[:, channel].mean(), pooled_normal_mean=right[:, channel].mean(),
                high_value_pair_win=wins[channel]))
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def pointwise_logits(model, attributes):
    """Exact no_graph forward. No edges, no neighbor states, no degree scaling."""
    import torch
    state = model.in_proj(attributes).relu()
    for layer in model.mp_layers:
        state = layer.update(state, torch.zeros_like(state))
    return model.pred(state).view(-1)


def pointwise_scores(model, attributes):
    import torch
    blocks = []
    device = model.in_proj.weight.device
    with torch.no_grad():
        for start in range(0, len(attributes), 2048):
            values = torch.as_tensor(attributes[start:start+2048], device=device, dtype=torch.float32)
            blocks.append(torch.sigmoid(pointwise_logits(model, values)).cpu().numpy())
    return np.concatenate(blocks)


def selected_units(geometry, args):
    layers, heads = geometry
    selected = range(layers) if args.llm_layers is None else args.llm_layers
    if args.channels:
        coordinates = [tuple(map(int, name.split(':'))) for name in args.channels]
    elif args.channel_unit == 'head':
        coordinates = [(layer, head) for layer in selected for head in range(heads)]
    else:
        coordinates = [(layer, -1) for layer in selected]
    units = [dict(name='all_swap', llm_layer=-1, llm_head=-1, operation='swap', columns=list(range(layers*heads)))]
    for layer, head in coordinates:
        if not 0 <= layer < layers or not -1 <= head < heads:
            raise ValueError('Node channel coordinates outside original LLM geometry')
        columns = [layer*heads+head] if head >= 0 else list(range(layer*heads, (layer+1)*heads))
        for operation in args.node_operations:
            units.append(dict(name=f'L{layer}_H{head}_{operation}', llm_layer=layer, llm_head=head,
                              operation=operation, columns=columns))
    if len({unit['name'] for unit in units}) != len(units):
        raise ValueError('Repeated node channel coordinates')
    return units


def changed_attributes(error, normal, unit):
    left, right = error.copy(), normal.copy()
    columns = unit['columns']
    if unit['operation'] == 'swap':
        left[:, columns], right[:, columns] = normal[:, columns], error[:, columns]
    else:
        left[:, columns], right[:, columns] = 0, 0
    return left, right


def intervention_rows(base_pairs, altered, pairs, threshold, unit, bootstrap):
    changed = paired_scores(altered, pairs, threshold)
    keys = ['id', 'source_id', 'tier', 'error_start', 'normal_start', 'region']
    result = base_pairs.merge(changed, on=keys, suffixes=('_base', '_changed'), validate='one_to_one')
    result['auc_delta'] = result.auroc_changed - result.auroc_base
    result['margin_delta'] = result.margin_changed - result.margin_base
    result['unit'] = unit['name']
    rows = []
    for region, group in result.groupby('region'):
        interval = source_interval(group, 'auc_delta', bootstrap)
        rows.append(dict(unit=unit['name'], llm_layer=unit['llm_layer'], llm_head=unit['llm_head'],
            operation=unit['operation'], region=region, pairs=len(group),
            base_auc=group.auroc_base.mean(), changed_auc=group.auroc_changed.mean(),
            margin_delta=group.margin_delta.mean(), error_hits_before=int(group.hits_base.sum()),
            error_hits_after=int(group.hits_changed.sum()), normal_fp_before=int(group.false_alarms_base.sum()),
            normal_fp_after=int(group.false_alarms_changed.sum()), **interval))
    return rows, result


def attribute_interventions(args, output, attributes, geometry, node, paired, pairs, threshold):
    from .model import load_checkpoint
    checkpoint = args.checkpoint or str(Path(args.root)/'node_only/checkpoint.pt')
    model, _ = load_checkpoint(checkpoint, args.device, 'in', args.edge_chunk)
    replay = pointwise_scores(model, attributes)
    replay_error = float(np.max(abs(replay-node.score.to_numpy())))
    if replay_error > 2e-5:
        raise ValueError(f'node_only pointwise replay mismatch {replay_error:.6g}; no attribution performed')
    error_indices, normal_indices = pair_indices(node, paired)
    error, normal = attributes[error_indices], attributes[normal_indices]
    base_pairs = paired_scores(node, pairs, threshold)
    units = selected_units(geometry, args)
    rows, changes, scores = [], [], {}
    for unit in tqdm(units, desc='node-only channel exchange', unit='block'):
        left, right = changed_attributes(error, normal, unit)
        predicted = pointwise_scores(model, np.concatenate([left, right]))
        error_score, normal_score = np.split(predicted, 2)
        if unit['name'] == 'all_swap':
            expected = np.r_[replay[normal_indices], replay[error_indices]]
            if not np.allclose(predicted, expected, atol=2e-5, rtol=0):
                raise ValueError('Full-vector exchange did not exchange the original node-only scores')
        altered = node.copy()
        altered.loc[error_indices, 'score'] = error_score
        altered.loc[normal_indices, 'score'] = normal_score
        unit_rows, per_pair = intervention_rows(base_pairs, altered, pairs, threshold, unit, args.bootstrap)
        rows.extend(unit_rows)
        changes.append(per_pair)
        scores[unit['name']] = predicted
    save_scores(output/'exchange_scores.npz', **scores)
    pd.DataFrame(rows).to_csv(output/'node_channel_effects.csv', index=False)
    pd.concat(changes, ignore_index=True).to_csv(output/'pair_channel_effects.csv.gz', index=False)
    write_json(output/'replay.json', dict(checkpoint=str(checkpoint), max_abs_error=replay_error,
        full_swap='passed', forward='pointwise: projection -> own-node updates with zero messages -> readout'))
    write_json(output/'units.json', units)


def run_node(args, output, pairs):
    from .node_plots import node_plots

    pairs = [pair for pair in pairs if pair['tier'] == args.pair_tier]
    if not pairs:
        raise ValueError('Node audit requires nonempty fixed normal/error pairs')
    tables, _, thresholds = aligned_models(args.root)
    node = tables['node_only']
    comparison = comparison_tokens(tables)
    paired = matched_tokens(comparison, pairs, args.pair_tier)
    attributes, geometry = load_node_attributes(args.root, args.prepared, node)
    error_indices, normal_indices = pair_indices(node, paired)
    error, normal = attributes[error_indices], attributes[normal_indices]
    stats, coverage = attribute_statistics(error, normal, paired, geometry[1], args.bootstrap)
    stats.to_csv(output/'node_attribute_differences.csv.gz', index=False)
    coverage.to_csv(output/'cohort_counts.csv', index=False)
    paired.to_csv(output/'paired_tokens.csv.gz', index=False)
    save_scores(output/'paired_node_attributes.npz', error=error, normal=normal,
                layers=np.asarray(geometry[0]), heads=np.asarray(geometry[1]))
    attribute_interventions(args, output, attributes, geometry, node, paired, pairs, thresholds['node_only']['value'])
    node_plots(stats, pd.read_csv(output/'node_channel_effects.csv'), output/'figures')
    write_json(output/'protocol.json', dict(model='node_only', pair_tier=args.pair_tier,
        raw_input='Only x: per-layer/head self-attention diagonal. No edge arrays are loaded.',
        axes='Original LLM layer/head. The node network mixes channels; GNN depth is a separate axis.',
        threshold=thresholds['node_only'], cohorts='all is primary; score-defined TP/FN groups are descriptive selection.',
        uncertainty='Equal-pair source means; exploratory source bootstrap, no multiple-testing claim.',
        attribution='Matched channel exchange/zeroing of the frozen node model, not native LLM intervention.',
        limitations=['Conditional swaps can be off-distribution.', 'Raw attribute contrast is not a model importance.',
                     'Full-vector exchange is an implementation check, not a scientific discovery.',
                     'No conclusion about prompt mass or self lock-in is produced by this node-only audit.']))
    print('Node-only input contrasts and exchange effects:', output, flush=True)
