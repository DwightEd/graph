"""Single-main entry for model-based internal reuse, not attention feature scoring."""

from pathlib import Path
import zlib

import numpy as np
from tqdm import tqdm

from .data import load_graph, load_predictions, read_json, write_json, save_scores


PROTOCOL = 'charm_internal_reuse_v1'


def file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def make_manifest(args, pairs, samples, checkpoint):
    records = {str(row['id']): row for row in samples}
    selected = []
    graphs = {}
    for number, pair in enumerate(pairs):
        identity = str(pair['id'])
        row = records[identity]
        if str(pair['source_id']) != str(row['source_id']):
            raise ValueError('Fixed pair and original prediction source differ')
        selected.append(dict(pair, id=identity, source_id=str(pair['source_id']),
                             file=f'{number:06d}.npz'))
        graph_path = Path(args.prepared) / 'graphs' / row['split'] / (identity + '.npz')
        graphs[identity] = file_stamp(graph_path)
    return dict(protocol=PROTOCOL, checkpoint=file_stamp(checkpoint), graphs=graphs,
                pairs=selected, random_repeats=args.lockin_random,
                subset=bool(args.lockin_pair_limit),
                thresholds=read_json(Path(args.root) / 'charm_in/threshold.json'))


def validate_pair_set(pairs):
    """Scientific boundary: disjoint roles and no duplicate normal tokens."""
    used = {}
    for pair in pairs:
        error, normal, length = pair['error_start'], pair['normal_start'], pair['length']
        if length < 1 or min(error, normal) < 0 or abs(error - normal) < length:
            raise ValueError('Invalid or overlapping paired windows')
        for role, start in (('error', error), ('normal', normal)):
            coordinates = set(range(start, start + length))
            key = str(pair['id']), role
            if used.setdefault(key, set()) & coordinates:
                raise ValueError('Overlapping windows would duplicate token denominators')
            used[key].update(coordinates)


def validate_graph(graph, sample, record):
    source, target = graph['edge_index']
    prompt = int(graph['prompt_length'])
    if np.any(source < 0) or np.any(source >= target) or np.any(target < prompt):
        raise ValueError('Exact local replay requires causal past-to-response edges')
    if graph['x'].shape[1] != int(graph['layers']) * int(graph['heads']):
        raise ValueError('Original LLM channel geometry mismatch')
    for key in ('gold', 'offsets', 'spans', 'response'):
        if not np.array_equal(record[key], sample[key]):
            raise ValueError('Original graph/prediction alignment mismatch: ' + key)


def pair_metadata(record, pair, threshold):
    length = pair['length']
    indices = np.stack([np.arange(pair[key], pair[key] + length)
                        for key in ('error_start', 'normal_start')])
    if indices.max() >= len(record['gold']):
        raise ValueError('Pair extends beyond the response')
    if not record['gold'][indices[0]].all() or record['gold'][indices[1]].any():
        raise ValueError('Locked error/normal pair does not match original labels')
    text = str(record['response'])
    tokens = np.asarray([[text[a:b] for a, b in record['offsets'][side]] for side in indices])
    return dict(tokens=indices, text=tokens, baseline_score=record['score'][indices],
                threshold=np.asarray(threshold),
                tail_cutoffs=np.quantile(record['score'], [.2, .8]))


def run_pair(model, graph, sample, record, base, pair, args, threshold, output):
    from .lockin_forward import pair_forward

    metadata = pair_metadata(record, pair, threshold)
    path = output / 'captures' / pair['file']
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            for key in ('baseline_score', 'tokens', 'text', 'threshold'):
                np.testing.assert_array_equal(saved[key], metadata[key])
        return
    seed = args.seed + zlib.crc32(f"{pair['id']}:{pair['error_start']}".encode())
    arrays, controls = pair_forward(model, graph, sample, base, pair, args.lockin_random, seed)
    from scipy.special import expit
    np.testing.assert_allclose(expit(arrays['logits'][:, 0]), metadata['baseline_score'], atol=2e-5, rtol=0)
    save_scores(path, **arrays, **metadata)
    write_json(output / 'controls' / (Path(pair['file']).stem + '.json'), controls)


def run_lockin(args, output, pairs):
    """No changes to model/inputs/scores; report-only never imports torch."""
    from .lockin_report import report

    output = Path(output)
    if args.lockin_stage == 'report':
        report(output, args.bootstrap)
        return
    selected = [pair for pair in pairs if pair['tier'] == args.pair_tier]
    if args.lockin_pair_limit:
        selected = selected[:args.lockin_pair_limit]
    if not selected:
        raise ValueError('No fixed pairs in requested tier; no easy replacements will be chosen')
    if args.lockin_random < 0 or args.lockin_pair_limit < 0:
        raise ValueError('Random repeats and pair limit must be nonnegative')
    validate_pair_set(selected)
    samples = load_predictions(Path(args.root) / 'charm_in/test')
    checkpoint = args.checkpoint or str(Path(args.root) / 'charm_in/checkpoint.pt')
    manifest = make_manifest(args, selected, samples, checkpoint)
    path = output / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Original checkpoint/graphs/pairs changed; use a new output directory')
    write_json(path, manifest)
    execute_pairs(args, output, manifest, samples, checkpoint)
    report(output, args.bootstrap)


def execute_pairs(args, output, manifest, samples, checkpoint):
    import torch
    from .model import load_checkpoint
    from .lockin_forward import factual_states

    model, _ = load_checkpoint(checkpoint, args.device, 'in', args.edge_chunk)
    threshold = float(manifest['thresholds']['value'])
    grouped = {}
    for pair in manifest['pairs']:
        grouped.setdefault(pair['id'], []).append(pair)
    replay = []
    progress = tqdm(total=len(manifest['pairs']), desc='trained-CHARM internal reuse', unit='pair')
    with torch.no_grad():
        for record in samples:
            identity = str(record['id'])
            if identity not in grouped:
                continue
            graph, sample = load_graph(record, args.prepared)
            validate_graph(graph, sample, record)
            base = factual_states(model, graph)
            score = torch.sigmoid(model.pred(base[-1]).view(-1))[int(graph['prompt_length']):].cpu().numpy()
            error = float(np.max(np.abs(score - record['score'])))
            if error > 2e-5:
                raise ValueError(f'Original full-score replay failed for {identity}: {error}')
            replay.append(dict(id=identity, max_abs_error=error))
            for pair in grouped[identity]:
                run_pair(model, graph, sample, record, base, pair, args, threshold, output)
                progress.update(1)
            del graph, sample, base
    progress.close()
    write_json(output / 'replay.json', replay)
