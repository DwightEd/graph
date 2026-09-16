"""顺序流程：提取固定表示 → 建参照 → 冻结评分；评价独立执行。"""

from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from ..offline_span.data import write_json
from .inputs import extract_sample, identity_record, list_samples, sample_channels, save_sample, validate_roster
from .operators import VARIANTS
from .reference import (apply_calibration, binary_spans, calibrate, fit_reference,
                        gather_embeddings, group_records, novelty_distance,
                        sample_reference_rows, smooth_scores, split_sources)


SCORES = (*VARIANTS, 'node_smooth')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_input_roster(root, samples):
    """续跑只核对输入身份和原attention文件时间/大小，不建立新数据集。"""
    roster = []
    for sample in samples:
        files = []
        for name in sample.cache_files:
            path = Path(name)
            stamp = path.stat()
            files.append([str(path.resolve()), stamp.st_size, stamp.st_mtime_ns])
        roster.append(dict(id=sample.response_id, source=sample.source_id,
                           task=sample.task, generator=sample.generator,
                           split=sample.split, files=files))
    path = root / 'inputs.json'
    if path.exists() and read_json(path) != roster:
        raise ValueError('Original input roster changed; use a new output')
    write_json(path, roster)


def prepare(args):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    root = args.output / 'features'
    root.mkdir(parents=True, exist_ok=True)
    check_input_roster(root, samples)
    records = []
    layout = None
    for number, sample in enumerate(tqdm(samples, desc='fixed graph', unit='answer')):
        path = root / f'{number:06d}.npz'
        if not (args.resume and path.is_file()):
            channels = sample_channels(sample, indexes[sample.split], args.layers, args.heads)
            extracted = extract_sample(sample, channels, vars(args))
            save_sample(path, sample, extracted)
        with np.load(path, allow_pickle=False) as archive:
            current = sorted(map(tuple, archive['channels'].tolist()))
            if not np.array_equal(archive['token_ids'], sample.token_ids):
                raise ValueError('Cached feature roster changed; select a new output')
        if layout is not None and current != layout:
            raise ValueError('Physical channel layout differs between answers')
        layout = current
        records.append(identity_record(sample, path.name))
    write_json(root / 'manifest.json', dict(records=records, channels=layout,
                                           labels_used=False, complete=True))
    return records


def load_scores(root, row, references):
    with np.load(root / row['file'], allow_pickle=False) as archive:
        coverage = archive['coverage'].copy()
        scores = {}
        for name in VARIANTS:
            values = archive[name]
            output = np.full(len(coverage), np.nan)
            output[coverage] = novelty_distance(values[coverage], references[name])
            scores[name] = output
    return scores, coverage


def group_calibration(root, rows, references, args):
    values = {name: [] for name in SCORES}
    sources = []
    for row in tqdm(rows, desc='unlabeled calibration', leave=False):
        scores, coverage = load_scores(root, row, references)
        scores['node_smooth'] = smooth_scores(scores['nodes'], args.local_window)
        for name in SCORES:
            values[name].extend(scores[name][coverage])
        sources.extend([row['source_id']] * int(coverage.sum()))
    if not sources:
        raise ValueError('No calibration tokens for this task/generator')
    return {name: calibrate(np.asarray(values[name]), sources, args.quantile) for name in SCORES}


def save_references(directory, references):
    arrays = {}
    for name, reference in references.items():
        for key, value in reference.items():
            arrays[f'{name}__{key}'] = value
    np.savez_compressed(directory / 'bank.npz', **arrays)


def load_references(directory):
    with np.load(directory / 'bank.npz', allow_pickle=False) as archive:
        return {name: {key: archive[f'{name}__{key}'] for key in ('bank', 'scale', 'center', 'neighbors')}
                for name in VARIANTS}


def fit(args):
    feature_root = args.output / 'features'
    records = read_json(feature_root / 'manifest.json')['records']
    roles = split_sources(records, args.seed)
    root = args.output / 'reference'
    root.mkdir(exist_ok=True)
    settings = dict(roles=roles, groups={}, labels_used=False)
    train = [row for row in records if row['split'] == 'train']
    for number, (group, rows) in enumerate(sorted(group_records(train).items())):
        fitting = [row for row in rows if roles[row['source_id']] == 'reference']
        calibration = [row for row in rows if roles[row['source_id']] == 'calibration']
        if not fitting or not calibration:
            raise ValueError(f'{group}: both reference and calibration sources are required')
        selected = sample_reference_rows(feature_root, fitting, args.bank_size, args.tokens_per_source, args.seed)
        arrays = gather_embeddings(feature_root, selected)
        references = {name: fit_reference(arrays[name], args.neighbors) for name in VARIANTS}
        directory = root / str(number)
        directory.mkdir(exist_ok=True)
        save_references(directory, references)
        controls = group_calibration(feature_root, calibration, references, args)
        settings['groups']['|'.join(group)] = dict(directory=str(number), calibration=controls,
                                                  bank_rows=selected, reference_sources=sorted({s for _, _, s in selected}))
    write_json(root / 'settings.json', settings)
    write_json(root / 'complete.json', dict(complete=True))


def score_answer(args, row, references, calibration, path):
    root = args.output / 'features'
    raw, coverage = load_scores(root, row, references)
    raw['node_smooth'] = smooth_scores(raw['nodes'], args.local_window)
    arrays = {}
    for name in SCORES:
        values = apply_calibration(raw[name], calibration[name])
        alarm = values > calibration[name]['threshold']
        arrays[name] = values
        arrays[name + '_spans'] = binary_spans(alarm & coverage)
    with np.load(root / row['file'], allow_pickle=False) as archive:
        for key in ('token_ids', 'offsets', 'diagnostics'):
            arrays[key] = archive[key]
        arrays['local_mass'] = archive['mean_attributes'][:, 1]
        arrays['attention_entropy'] = archive['mean_attributes'][:, 3]
    arrays['position'] = np.arange(row['tokens'], dtype=float)
    for name in ('local_mass', 'attention_entropy', 'position'):
        arrays[name][~coverage] = np.nan
    arrays['coverage'] = coverage
    arrays['prompt_length'] = row['prompt_length']
    arrays['record_json'] = json.dumps(dict(row, file=path.name))
    if len(arrays['offsets']) == 0:
        del arrays['offsets']
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def score(args):
    records = read_json(args.output / 'features/manifest.json')['records']
    records = [row for row in records if row['split'] == 'test']
    settings = read_json(args.output / 'reference/settings.json')
    output = args.output / 'predictions'
    output.mkdir(exist_ok=True)
    references = {}
    saved = []
    for number, row in enumerate(tqdm(records, desc='score all controls', unit='answer')):
        group = row['task'] + '|' + row['generator']
        configured = settings['groups'][group]
        if group not in references:
            directory = args.output / 'reference' / configured['directory']
            references[group] = load_references(directory)
        path = output / f'{number:06d}.npz'
        if not (args.resume and path.is_file()):
            score_answer(args, row, references[group], configured['calibration'], path)
        saved.append(dict(row, file=path.name))
    write_json(output / 'freeze.json', dict(records=saved, complete=True, labels_used=False))


def inspect(args):
    samples, indexes = list_samples(args)
    validate_roster(samples)
    for split in ('train', 'test'):
        subset = [sample for sample in samples if sample.split == split]
        first = subset[0]
        channels = sample_channels(first, indexes[split], args.layers, args.heads)
        embeddings, coverage, layout, diagnostics, _ = extract_sample(first, channels, vars(args))
        print(json.dumps(dict(split=split, answers=len(subset), first_id=first.response_id,
                              tasks=sorted({sample.task for sample in subset}), channels=len(layout),
                              coverage=float(coverage.mean()), dimensions=args.dimensions,
                              observation_scope='first answer per split', neural_training=False,
                              mean_moved_mass=float(diagnostics[:, 2].mean()))), flush=True)
