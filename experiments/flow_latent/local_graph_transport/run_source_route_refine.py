"""Freeze a source/route/graph refinement on cached QA fit, pilot and DEV.

No model is loaded or natural-label array opened during fit or scoring.
Run the separate evaluator only after the per-phase FREEZE exists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .data import PACKS
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_unlabeled import answer_inputs, weighted_reference_threshold
from .source_route_refine import (METHODS, PRIMARY, ROUTE_WEIGHT, edge_reference,
                                  fit_fusion_reference, score_answer, unaries,
                                  weighted_median)
from .unlabeled import equal_source_weights


DEFAULT_OUTPUT = Path('outputs/source_route_refine_20261009')
PILOT_IDS = ('12219', '12216', '12297', '12294', '15604', '15600', '11907', '11904')
PACK_FIELDS = ('token_id', 'target', 'source_index', 'answer_index', 'unit_index')
LEGACY = OUTPUT / 'unlabeled'


def read_unlabeled_pack(split):
    with np.load(PACKS / f'QA_{split}.npz') as packed:
        pack = {name: packed[name] for name in PACK_FIELDS}
    metadata = json.loads((PACKS / f'QA_{split}.json').read_text())
    return pack, metadata


def selected_pack(rows):
    """Keep exact original token targets, rebuilding only contiguous output offsets."""
    parts = {name: [] for name in PACK_FIELDS}
    records = []
    sources = {source: index for index, source in
               enumerate(sorted({row['source_id'] for row, _ in rows}))}
    offset = 0
    for index, (record, pack) in enumerate(rows):
        region = slice(record['packed_start'], record['packed_stop'])
        count = record['packed_stop'] - record['packed_start']
        records.append(dict(record, packed_start=offset, packed_stop=offset + count))
        for name in PACK_FIELDS:
            values = pack[name][region]
            if name == 'answer_index':
                values = np.full(count, index, dtype=np.int64)
            elif name == 'source_index':
                # Train/test packs number sources independently; combine by actual ID.
                values = np.full(count, sources[record['source_id']], dtype=np.int64)
            parts[name].append(values)
        offset += count
    if not records:
        raise ValueError('Empty scoring population')
    return ({name: np.concatenate(value) for name, value in parts.items()},
            dict(task='QA', records=records, sources=sorted({r['source_id'] for r in records})))


def verify_legacy():
    frozen = json.loads((LEGACY / 'FREEZE.json').read_text())
    names = ('reference.npz', 'train_ranks.npz', 'train_scores.npz', 'test_scores.npz')
    hashes = {name: file_hash(LEGACY / name) for name in names}
    for name, actual in hashes.items():
        if actual != frozen['hashes'][name]:
            raise ValueError(f'Legacy frozen artifact changed: {name}')
    for name, expected in frozen['pack_hashes'].items():
        if file_hash(PACKS / name) != expected:
            raise ValueError(f'Legacy pack changed: {name}')
    return hashes


def snapshot(directory):
    target = directory / 'code_snapshot'
    target.mkdir()
    names = ('source_route_refine.py', 'run_source_route_refine.py',
             'source_route_refine_eval.py', 'test_source_route_refine.py',
             'test_source_route_refine_eval.py',
             'smooth.py', 'unlabeled.py', 'run_unlabeled.py', 'run_capture.py', 'data.py')
    for name in names:
        shutil.copy2(Path(__file__).with_name(name), target / name)
    return {name: file_hash(target / name) for name in names}


def update_digest(digest, record, values, attention):
    digest.update(record['id'].encode())
    for name, value in (*values.items(), ('local_attention', attention)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())


def score_records(reference, fusion, delta, pack, metadata, legacy_scores):
    scores = {name: np.full(len(pack['target']), np.nan) for name in METHODS}
    ranks = {}
    diagnostics = []
    digest = hashlib.sha256()
    baseline_difference = 0.
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, pack)
        update_digest(digest, record, values, attention)
        result = score_answer(reference, fusion, delta, values, attention)
        for name, value in result['scores'].items():
            scores[name][region] = value[targets]
        for name, value in result['ranks'].items():
            ranks.setdefault(name, np.full(len(pack['target']), np.nan))[region] = value[targets]
        expected = legacy_scores[record['id']]
        baseline_difference = max(baseline_difference,
            float(np.max(np.abs(scores['old_native'][region] - expected))))
        if baseline_difference > 1e-8:
            raise ValueError(f"{record['id']}: old baseline changed by {baseline_difference}")
        diagnostics.append(dict(id=record['id'], source_id=record['source_id'],
            full_tokens=record['tokens'], valid_tokens=len(targets), methods=result['diagnostics']))
        if (index + 1) % 100 == 0:
            print(f"SCORE {index + 1}/{len(metadata['records'])}", flush=True)
    if not all(np.isfinite(value).all() for value in (*scores.values(), *ranks.values())):
        raise ValueError('Incomplete/nonfinite full-population scores')
    return scores, ranks, dict(answers=diagnostics,
        consumed_unlabeled_input_sha256=digest.hexdigest(),
        old_native_max_difference=baseline_difference, offline=True)


def legacy_answer_scores(train_meta, test_meta):
    result = {}
    for split, metadata in (('train', train_meta), ('test', test_meta)):
        with np.load(LEGACY / f'{split}_scores.npz') as archived:
            values = archived['source_route_native_huber']
        for record in metadata['records']:
            result[record['id']] = values[record['packed_start']:record['packed_stop']]
    return result


def freeze(directory, phase, diagnostics, code_hashes):
    names = [f'{phase}_{name}' for name in ('scores.npz', 'ranks.npz', 'pack.npz',
                                           'metadata.json', 'diagnostics.json')]
    names += ['REFINE_REFERENCE.json', 'fusion_reference.npz', 'scalar_reference.npz']
    frozen = dict(status='all_predictions_frozen', phase=phase, primary=PRIMARY,
        methods=list(METHODS), hashes={name: file_hash(directory / name) for name in names},
        code_hashes=code_hashes, diagnostics=diagnostics,
        natural_labels_used_for_fit_or_score=False, historical_population_exposure=True,
        pack_hashes={f'QA_{split}.{suffix}': file_hash(PACKS / f'QA_{split}.{suffix}')
                     for split in ('train', 'test') for suffix in ('npz', 'json')},
        scope='offline full-answer graph; Llama3.1 observer of mixed-generator answers')
    write_json(directory / f'{phase}_FREEZE.json', frozen)


def fit(args):
    started = time.time()
    legacy_hashes = verify_legacy()
    train, train_meta = read_unlabeled_pack('train')
    _, test_meta = read_unlabeled_pack('test')
    records = [row for row in train_meta['records'] if row['partition'] == 'fit']
    if args.fit_limit is not None:
        records = records[:args.fit_limit]
    selected, metadata = selected_pack([(row, train) for row in records])
    args.output.mkdir(parents=True, exist_ok=False)
    code_hashes = snapshot(args.output)
    shutil.copy2(LEGACY / 'reference.npz', args.output / 'scalar_reference.npz')
    with np.load(args.output / 'scalar_reference.npz') as loaded:
        reference = dict(loaded)
    fit_indices = np.concatenate([np.arange(row['packed_start'], row['packed_stop']) for row in records])
    with np.load(LEGACY / 'train_ranks.npz') as archived:
        local, full = archived['source_local'][fit_indices], archived['source_full'][fit_indices]
    fusion = fit_fusion_reference(local, full, selected['source_index'])
    np.savez(args.output / 'fusion_reference.npz', **fusion)
    differences, weights, source_groups = [], [], []
    identities = {source: index for index, source in enumerate(metadata['sources'])}
    digest = hashlib.sha256()
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(CACHE, OUTPUT / 'capture', record, selected)
        update_digest(digest, record, values, attention)
        fields, ranks = unaries(reference, fusion, values)
        np.testing.assert_allclose(ranks['source_local'][targets], local[region], atol=0, rtol=0)
        np.testing.assert_allclose(ranks['source_full'][targets], full[region], atol=0, rtol=0)
        difference, weight = edge_reference(fields['old'], attention)
        differences.append(difference)
        weights.append(weight)
        source_groups.append(np.full(len(weight), identities[record['source_id']], dtype=np.int32))
        if (index + 1) % 100 == 0:
            print(f"FIT EDGES {index + 1}/{len(records)}", flush=True)
    differences, weights, source_groups = map(np.concatenate, (differences, weights, source_groups))
    source_mass = np.bincount(source_groups, weights=weights, minlength=len(identities))
    if (source_mass <= 0).any():
        raise ValueError('A fit source has no positive edge mass; no silent exclusion allowed')
    weights = weights / source_mass[source_groups] / len(identities)
    delta = weighted_median(differences, weights)
    print(f'FIT DELTA {delta:.10f}, edges={len(differences)}', flush=True)
    predictions, ranks, diagnostics = score_records(reference, fusion, delta, selected, metadata,
        legacy_answer_scores(train_meta, test_meta))
    thresholds = {name: weighted_reference_threshold(value, equal_source_weights(selected['source_index']))
                  for name, value in predictions.items()}
    write_json(args.output / 'REFINE_REFERENCE.json', dict(primary=PRIMARY,
        methods=list(METHODS), delta=delta, delta_rule='equal-source normalized-edge-mass median of old unary differences',
        delta_fallback=None, penalty=.5, route_weight=ROUTE_WEIGHT, thresholds=thresholds,
        threshold_rule='equal-source mixed-fit-token 95th percentile; strict >; not normal FPR',
        fit_answers=len(records), fit_sources=len(identities), fit_valid_tokens=len(selected['target']),
        complete_fit=args.fit_limit is None, software_witness=args.fit_limit is not None,
        source_edge_mass_min=float(source_mass.min()), fit_edges=len(differences),
        fit_delta_input_sha256=digest.hexdigest(), legacy_hashes=legacy_hashes,
        natural_labels_used=False, primary_label_selected=False,
        capture_protocol_sha256=file_hash(OUTPUT / 'CAPTURE_PROTOCOL.json'),
        wall_seconds=time.time() - started))
    np.savez(args.output / 'fit_scores.npz', **predictions)
    np.savez(args.output / 'fit_ranks.npz', **ranks)
    np.savez(args.output / 'fit_pack.npz', **selected)
    write_json(args.output / 'fit_metadata.json', metadata)
    write_json(args.output / 'fit_diagnostics.json', diagnostics)
    freeze(args.output, 'fit', diagnostics['old_native_max_difference'], code_hashes)
    print(f'FIT FROZEN answers={len(records)} seconds={time.time() - started:.2f}', flush=True)


def verify_reference(directory):
    frozen = json.loads((directory / 'fit_FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Fit frozen artifact changed: {name}')
    for name, expected in frozen['code_hashes'].items():
        if file_hash(directory / 'code_snapshot' / name) != expected:
            raise ValueError(f'Fit snapshot changed: {name}')
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Executing code differs from fit snapshot: {name}')
    for name, expected in frozen['pack_hashes'].items():
        if file_hash(PACKS / name) != expected:
            raise ValueError(f'Fit official pack changed: {name}')
    return frozen


def score(args):
    started = time.time()
    frozen = verify_reference(args.output)
    config = json.loads((args.output / 'REFINE_REFERENCE.json').read_text())
    if not config['complete_fit']:
        raise ValueError('Incomplete software-witness reference cannot score scientific cohorts')
    train, train_meta = read_unlabeled_pack('train')
    test, test_meta = read_unlabeled_pack('test')
    all_rows = [(row, train) for row in train_meta['records']] + [(row, test) for row in test_meta['records']]
    if args.phase == 'pilot':
        lookup = {row['id']: (row, pack) for row, pack in all_rows}
        rows = [lookup[identifier] for identifier in PILOT_IDS]
    else:
        rows = [(row, train) for row in train_meta['records'] if row['partition'] == 'dev']
    selected, metadata = selected_pack(rows)
    metadata.update(phase=args.phase, fit_overlap_ids=[r['id'] for r in metadata['records'] if r['partition'] == 'fit'])
    if (args.output / f'{args.phase}_scores.npz').exists():
        raise FileExistsError('Preserve existing predictions; do not overwrite a phase')
    with np.load(args.output / 'scalar_reference.npz') as loaded:
        reference = dict(loaded)
    with np.load(args.output / 'fusion_reference.npz') as loaded:
        fusion = dict(loaded)
    predictions, ranks, diagnostics = score_records(reference, fusion, config['delta'], selected,
        metadata, legacy_answer_scores(train_meta, test_meta))
    np.savez(args.output / f'{args.phase}_scores.npz', **predictions)
    np.savez(args.output / f'{args.phase}_ranks.npz', **ranks)
    np.savez(args.output / f'{args.phase}_pack.npz', **selected)
    write_json(args.output / f'{args.phase}_metadata.json', metadata)
    write_json(args.output / f'{args.phase}_diagnostics.json', diagnostics)
    freeze(args.output, args.phase, diagnostics['old_native_max_difference'], frozen['code_hashes'])
    print(f'{args.phase.upper()} FROZEN answers={len(rows)} seconds={time.time() - started:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'score'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--phase', choices=('pilot', 'dev'), default='pilot')
    parser.add_argument('--fit-limit', type=int, help='Software witness only; rejected for scientific scoring')
    args = parser.parse_args()
    torch.set_num_threads(4)
    (fit if args.stage == 'fit' else score)(args)


if __name__ == '__main__':
    main()
