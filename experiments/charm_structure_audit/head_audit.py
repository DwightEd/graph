"""Locate which ORIGINAL LLM channels the frozen CHARM detector uses.

Channel zeroing changes detector inputs, not the generating LLM. Union edges,
normalization and original thresholds stay fixed. Sensitivity is not causality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import load_graph, load_predictions, read_json, read_tables, save_scores, token_frame, write_json
from .evaluate import metrics, paired_scores, source_interval
from .positions import annotate
from .ablations import groups


def channel_units(layers, heads, args):
    selected_layers = args.llm_layers if args.llm_layers is not None else range(layers)
    if args.channels:
        coordinates = [tuple(map(int, name.split(':'))) for name in args.channels]
    elif args.channel_unit == 'head':
        coordinates = [(layer, head) for layer in selected_layers for head in range(heads)]
    else:
        coordinates = [(layer, -1) for layer in selected_layers]
    units = []
    for layer, head in coordinates:
        if not 0 <= layer < layers or not -1 <= head < heads:
            raise ValueError('Requested LLM layer/head is outside this observer geometry')
        columns = [layer * heads + head] if head >= 0 else list(range(layer * heads, (layer + 1) * heads))
        for site in args.channel_sites:
            for operation in args.channel_operations:
                if operation != 'zero' and site != 'edge':
                    continue
                name = f'{site}_L{layer}' + (f'_H{head}' if head >= 0 else '')
                name += '_' + operation
                units.append(dict(name=name, site=site, operation=operation, llm_layer=layer, llm_head=head, columns=columns))
    if not units:
        raise ValueError('Coupled/independent permutations apply only to edge channels')
    if len({unit['name'] for unit in units}) != len(units):
        raise ValueError('Repeated channel intervention names')
    return units


def mask_channels(graph, columns, site):
    """Zero one fixed input block. Do not rebuild the union graph or degrees."""
    result = dict(graph)
    counts = dict(changed_node_cells=0, changed_edge_cells=0)
    for field, where in (('x', 'node'), ('edge_attr', 'edge')):
        if site in (where, 'both'):
            values = graph[field].copy()
            counts['changed_' + where + '_cells'] = int(np.count_nonzero(values[:, columns]))
            values[:, columns] = 0
            result[field] = values
    return result, counts


def alter_channels(graph, unit, seed):
    if unit['operation'] == 'zero':
        return mask_channels(graph, unit['columns'], unit['site'])
    rng = np.random.default_rng(seed)
    columns = unit['columns']
    values = graph['edge_attr'].copy()
    for indices in groups(graph):
        if unit['operation'] == 'coupled':
            values[np.ix_(indices, columns)] = graph['edge_attr'][np.ix_(rng.permutation(indices), columns)]
        else:
            for channel in columns:
                values[indices, channel] = graph['edge_attr'][rng.permutation(indices), channel]
    changed = int(np.count_nonzero(values[:, columns] != graph['edge_attr'][:, columns]))
    zeros = int((~np.any(values > 0, axis=1)).sum())
    return dict(graph, edge_attr=values), dict(changed_node_cells=0, changed_edge_cells=changed, zero_edge_slots=zeros)


def capture_channels(args, output, samples):
    import torch
    import zlib
    from .model import load_checkpoint, degree

    checkpoint = args.checkpoint or str(Path(args.root) / 'charm_in' / 'checkpoint.pt')
    model, _ = load_checkpoint(checkpoint, args.device, 'in', args.edge_chunk)
    units, geometry, changes = None, None, []
    for record in tqdm(samples, desc='frozen detector channel masks', unit='answer'):
        graph, sample = load_graph(record, args.prepared)
        current = int(graph['layers']), int(graph['heads'])
        if geometry is None:
            geometry, units = current, channel_units(*current, args)
        if current != geometry:
            raise ValueError('One channel sweep must use one LLM observer geometry')
        if any(not np.array_equal(record[k], sample[k]) for k in ('gold', 'offsets', 'spans')):
            raise ValueError('Graph and original prediction coordinates differ')
        p = int(graph['prompt_length'])
        divisor = degree(graph, 'in')
        with torch.no_grad():
            original = torch.sigmoid(model(graph)[p:]).cpu().numpy()
        if not np.allclose(original, record['score'], rtol=0, atol=2e-5):
            raise ValueError('Checkpoint does not replay the original CHARM scores')
        path = output / 'scores' / (str(record['id']) + '.npz')
        scores = {'full': original}
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                scores.update({unit['name']: saved[unit['name']] for unit in units})
                if not np.allclose(saved['full'], original, atol=2e-5, rtol=0):
                    raise ValueError('Saved channel results refer to a different baseline')
        for unit in tqdm(units, desc=str(record['id']) + ' channels', leave=False, unit='mask'):
            changed, count = alter_channels(graph, unit, args.seed + zlib.crc32(str(record['id']).encode()))
            changes.append(dict(id=str(record['id']), name=unit['name'], **count))
            if unit['name'] not in scores:
                with torch.no_grad():
                    scores[unit['name']] = torch.sigmoid(model(changed, divisor=divisor)[p:]).cpu().numpy()
            del changed
        save_scores(path, **scores)
    pd.DataFrame(changes).to_csv(output / 'changed_cells.csv', index=False)
    write_json(output / 'units.json', units)
    return units


def compare_unit(base, scores, pairs, threshold, unit, bootstrap):
    control = base.copy()
    control['score'] = scores
    control['predicted'] = (scores > threshold).astype(int)
    baseline = metrics(base.gold, base.score, threshold)
    changed = metrics(control.gold, scores, threshold)
    result = {key: unit[key] for key in ('name', 'site', 'operation', 'llm_layer', 'llm_head')}
    result.update(changed)
    result.update({metric + '_delta': changed[metric] - baseline[metric]
                   for metric in ('auroc', 'ap', 'recall', 'fpr') if baseline[metric] is not None})
    original_pairs = paired_scores(base, pairs, threshold)
    altered_pairs = paired_scores(control, pairs, threshold)
    result.update(matched_pairs=0, matched_auc_delta=None, matched_margin_delta=None)
    pair_changes = pd.DataFrame()
    if not original_pairs.empty:
        keys = ['id', 'source_id', 'tier', 'error_start', 'normal_start', 'region']
        pair_changes = original_pairs.merge(altered_pairs, on=keys, suffixes=('_base', '_control'), validate='one_to_one')
        pair_changes['auc_delta'] = pair_changes.auroc_control - pair_changes.auroc_base
        pair_changes['margin_delta'] = pair_changes.margin_control - pair_changes.margin_base
        pair_changes['name'] = unit['name']
        whole = pair_changes[pair_changes.region == 'all']
        result.update(matched_pairs=len(whole), matched_auc_delta=float(whole.auc_delta.mean()),
                      matched_margin_delta=float(whole.margin_delta.mean()))
        result.update({'paired_source_' + key: value for key, value in source_interval(whole, 'auc_delta', bootstrap).items()})
    return result, pair_changes, role_changes(base, scores, threshold, unit['name'])


def role_changes(base, scores, threshold, name):
    """Signed changes for every role, not just the examples whose score fell."""
    error = base.gold.astype(bool).to_numpy()
    first = error & (base.span_index == 0) & (base.offset == 0)
    onset = error & (base.offset == 0)
    delta = scores - base.score.to_numpy()
    original_alarm = base.score.to_numpy() > threshold
    altered_alarm = scores > threshold
    rows = []
    for role, mask in (('first_error', first), ('later_onset', onset & ~first),
                       ('continuation', error & ~onset), ('normal', ~error)):
        chosen = np.asarray(mask, bool)
        rows.append(dict(name=name, role=role, tokens=int(chosen.sum()),
            mean_delta=float(delta[chosen].mean()) if chosen.any() else None,
            lost_alarms=int((chosen & original_alarm & ~altered_alarm).sum()),
            gained_alarms=int((chosen & ~original_alarm & altered_alarm).sum())))
    return rows


def compare_pairing_operations(frame, output, bootstrap):
    if frame.empty:
        return
    keys = ['id', 'source_id', 'tier', 'error_start', 'normal_start', 'region', 'llm_layer', 'llm_head']
    independent = frame[frame.operation == 'independent']
    coupled = frame[frame.operation == 'coupled']
    if independent.empty or coupled.empty:
        return
    both = independent.merge(coupled, on=keys, suffixes=('_independent', '_coupled'), validate='one_to_one')
    both['auc_delta'] = both.auroc_control_independent - both.auroc_control_coupled
    both['margin_delta'] = both.margin_control_independent - both.margin_control_coupled
    rows = []
    for (layer, head, region), group in both.groupby(['llm_layer', 'llm_head', 'region']):
        rows.append(dict(llm_layer=layer, llm_head=head, region=region, pairs=len(group),
            margin_delta=float(group.margin_delta.mean()), **source_interval(group, 'auc_delta', bootstrap)))
    pd.DataFrame(rows).to_csv(output / 'independent_minus_coupled.csv', index=False)


def run_heads(args, output, pairs):
    from .visualize import head_plots

    directory = Path(args.root) / 'charm_in'
    samples = load_predictions(directory / 'test')
    threshold = float(read_json(directory / 'threshold.json')['value'])
    _, spans = read_tables(directory / 'test')
    base = pd.concat([token_frame(sample, sample['score'], threshold) for sample in samples], ignore_index=True)
    base = annotate(base, spans)
    units = capture_channels(args, output, samples)
    pairs = [pair for pair in pairs if pair['tier'] == args.pair_tier]
    sources = {str(s['id']): str(s['source_id']) for s in samples}
    if any(sources[str(p['id'])] != str(p['source_id']) for p in pairs):
        raise ValueError('Pair source identity differs from the original predictions')
    effects, pair_rows, roles = [], [], []
    for unit in tqdm(units, desc='channel score attribution', unit='unit'):
        blocks = []
        for sample in samples:
            with np.load(output / 'scores' / (str(sample['id']) + '.npz'), allow_pickle=False) as saved:
                blocks.append(saved[unit['name']])
        result, changes, role_rows = compare_unit(base, np.concatenate(blocks), pairs, threshold, unit, args.bootstrap)
        effects.append(result)
        pair_rows.append(changes)
        changes['llm_layer'] = unit['llm_layer']
        changes['llm_head'] = unit['llm_head']
        changes['operation'] = unit['operation']
        roles.extend(role_rows)
    frame = pd.DataFrame(effects)
    frame.to_csv(output / 'channel_effects.csv', index=False)
    pd.DataFrame(roles).to_csv(output / 'role_changes.csv', index=False)
    all_pairs = pd.concat(pair_rows, ignore_index=True)
    all_pairs.to_csv(output / 'pair_effects.csv.gz', index=False)
    compare_pairing_operations(all_pairs, output, args.bootstrap)
    head_plots(frame, output / 'figures')
    write_json(output / 'protocol.json', dict(threshold=threshold, units=len(units), pair_tier=args.pair_tier,
        interpretation='Frozen CHARM input-channel sensitivity, NOT ablating the original LLM heads',
        retained='same union edges, original degrees, original threshold, same matched normal windows',
        limitations=['Zeroing can be out of distribution; no causal generation mechanism is inferred.',
                     'No head is selected automatically using test performance.',
                     'Several source-bootstrap intervals are exploratory, not multiplicity-corrected.',
                     'A layer block changes all its heads together; individual heads are a separate sweep.',
                     'Independent vs coupled permutation does not alone prove semantic head cooperation; nonlinear edge-vector combinations can also matter.']))
