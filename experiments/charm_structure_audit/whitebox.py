"""Replay -> exact local circuit -> paired path attribution -> intervention.

Only the trained node_only checkpoint and original x enter the computation.
The source/label metadata are used for alignment and fixed diagnostic pairing.
"""

from pathlib import Path
import json
import zlib

import numpy as np
from tqdm import tqdm

from .data import load_predictions, read_json, write_json, save_scores
from .lockin import file_stamp, validate_pair_set, pair_metadata


def node_input(record, prepared):
    """Read named NPZ members only: edge tensors are never materialized."""
    path = Path(prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
    with np.load(path, allow_pickle=False) as saved:
        identity = json.loads(str(saved['record_json']))
        geometry = int(saved['layers']), int(saved['heads'])
        values = saved['x'][int(saved['prompt_length']):].copy()
        for key in ('gold', 'offsets', 'spans', 'response'):
            if not np.array_equal(record[key], saved[key]):
                raise ValueError('Node/prediction alignment differs: ' + key)
    if str(identity['source_id']) != str(record['source_id']) or str(identity['id']) != str(record['id']):
        raise ValueError('Node input belongs to a different answer/source')
    if values.shape != (len(record['gold']), geometry[0] * geometry[1]) or not np.isfinite(values).all():
        raise ValueError('Node input is not a finite original LLM layer/head matrix')
    return values, geometry


def explain_pair(model, error, normal, geometry, args, seed):
    from .whitebox_math import evaluate_nodes, integrate_difference, hybrid_logits, matched_random_masks
    import torch

    left, right = evaluate_nodes(model, error), evaluate_nodes(model, normal)
    integral = integrate_difference(model, normal, error, args.wb_max_points)
    arrays = dict(error_x=error, normal_x=normal, **integral)
    for name in ('logits', 'beta', 'intercept', 'contributions', 'gates'):
        arrays[name] = torch.stack((left[name], right[name]))
    for endpoint, values in ((left, error), (right, normal)):
        rebuilt = (endpoint['beta'] * values).sum(dim=-1) + endpoint['intercept']
        torch.testing.assert_close(rebuilt, endpoint['logits'], atol=1e-8, rtol=1e-7)
    layer_removed, layer_added = [], []
    for layer in range(geometry[0]):
        mask = torch.zeros_like(error, dtype=torch.bool)
        mask[:, layer * geometry[1]:(layer + 1) * geometry[1]] = True
        removed, added = hybrid_logits(model, error, normal, mask)
        layer_removed.append(removed)
        layer_added.append(added)
    arrays['layer_removed'] = torch.stack(layer_removed)
    arrays['layer_added'] = torch.stack(layer_added)
    selected, random, exchangeable = matched_random_masks(
        integral['attribution'], error - normal, geometry[1], args.wb_budget, args.wb_random, seed)
    masks = torch.cat((selected[None], random))
    outcomes = [hybrid_logits(model, error, normal, mask) for mask in masks]
    arrays.update(selected_mask=selected, random_masks=random, exchangeable=torch.as_tensor(exchangeable),
                  selected_removed=torch.stack([x[0] for x in outcomes]),
                  selected_added=torch.stack([x[1] for x in outcomes]))
    return {name: value.detach().cpu().numpy() for name, value in arrays.items()}


def manifest_for(args, selected, samples, checkpoint):
    lookup = {str(row['id']): row for row in samples}
    nodes = {}
    entries = []
    for number, pair in enumerate(selected):
        record = lookup[str(pair['id'])]
        if str(pair['source_id']) != str(record['source_id']):
            raise ValueError('Pair source differs from node prediction source')
        path = Path(args.prepared) / 'graphs' / record['split'] / (str(record['id']) + '.npz')
        nodes[str(record['id'])] = file_stamp(path)
        entries.append(dict(pair, file=f'{number:06d}.npz'))
    return dict(protocol='node_whitebox_v1', checkpoint=file_stamp(checkpoint), nodes=nodes,
                pairs=entries, model='node_only', numerical_max_points=args.wb_max_points,
                budget=args.wb_budget, random_repeats=args.wb_random, seed=args.seed,
                input_alignment='post_token_i_to_label_i',
                intervention_scope='paired contrast in frozen classifier; not a standalone one-layer detector')


def execute(args, output, manifest, samples, checkpoint):
    import torch
    from .model import load_checkpoint
    from .whitebox_math import evaluate_nodes

    model, _ = load_checkpoint(checkpoint, args.device)
    model.requires_grad_(False)
    threshold = float(read_json(Path(args.root) / 'node_only/threshold.json')['value'])
    by_answer = {}
    for pair in manifest['pairs']:
        by_answer.setdefault(str(pair['id']), []).append(pair)
    replays, geometry = [], None
    for record in tqdm(samples, desc='whitebox node replay', unit='answer'):
        identity = str(record['id'])
        if identity not in by_answer:
            continue
        values, current = node_input(record, args.prepared)
        if geometry is not None and geometry != current:
            raise ValueError('Mixed LLM channel geometries in one whitebox run')
        geometry = current
        write_json(output / 'geometry.json', dict(layers=geometry[0], heads=geometry[1]))
        # Original float32 replay precedes float64 numerical accounting.
        model.float()
        inputs = torch.as_tensor(values, device=args.device, dtype=torch.float32)
        predicted = evaluate_nodes(model, inputs)['logits'].sigmoid().cpu().numpy()
        error = float(abs(predicted - record['score']).max())
        if error > 2e-5:
            raise ValueError(f'node_only replay failed for {identity}: {error}; attribution stopped')
        replays.append(dict(id=identity, max_abs_score_error=error))
        model.double()
        inputs = inputs.double()
        for pair in tqdm(by_answer[identity], desc=identity + ' pairs', leave=False):
            run_pair(args, output, record, pair, model, inputs, geometry, threshold)
    write_json(output / 'replay.json', replays)
    write_json(output / 'geometry.json', dict(layers=geometry[0], heads=geometry[1]))


def run_pair(args, output, record, pair, model, inputs, geometry, threshold):
    metadata = pair_metadata(record, pair, threshold)
    path = output / 'captures' / pair['file']
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for key in ('baseline_score', 'tokens', 'text', 'threshold'):
                np.testing.assert_array_equal(saved[key], metadata[key])
        return
    indices = metadata['tokens']
    seed = args.seed + zlib.crc32(f"{pair['id']}:{pair['error_start']}".encode())
    arrays = explain_pair(model, inputs[indices[0]], inputs[indices[1]], geometry, args, seed)
    from scipy.special import expit
    np.testing.assert_allclose(expit(arrays['logits']), metadata['baseline_score'], atol=2e-5, rtol=0)
    save_scores(path, **arrays, **metadata)


def run_whitebox(args, output, pairs):
    from .whitebox_report import report

    output = Path(output)
    if args.wb_stage == 'report':
        report(output, args.bootstrap)
        return
    if args.wb_max_points < 32 or args.wb_max_points & (args.wb_max_points - 1):
        raise ValueError('--wb-max-points must be a power of two >= 32')
    if args.wb_budget < 1 or args.wb_random < 1:
        raise ValueError('A positive channel budget and at least one random control are required')
    selected = [pair for pair in pairs if pair['tier'] == args.pair_tier]
    if not selected:
        raise ValueError('No fixed pairs in requested tier; no substitute pairs are selected')
    validate_pair_set(selected)
    samples = load_predictions(Path(args.root) / 'node_only/test')
    checkpoint = args.checkpoint or str(Path(args.root) / 'node_only/checkpoint.pt')
    manifest = manifest_for(args, selected, samples, checkpoint)
    path = output / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Whitebox input/config changed; use a new --output')
    write_json(path, manifest)
    execute(args, output, manifest, samples, checkpoint)
    report(output, args.bootstrap)
