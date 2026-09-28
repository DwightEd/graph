"""Iteration two: compare adoption among similar native reading patterns."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from .density import distances, fit_scale, standardize, local_ratios
from .features import BLOCKS
from .score import reference_indices, reference_matrix, block_distances, OLD

NEW = ('association', 'adoption_given_reading', 'reading_given_adoption',
       'association_fused', 'association_gate', 'transition_gate')


def prepare(first, output):
    output.mkdir(exist_ok=False)
    manifest = read_json(first / 'manifest.json')
    write_json(output / 'manifest.json', dict(manifest, previous=str(first.resolve())))
    for row in manifest['records']:
        directory = output / row['key']
        directory.mkdir()
        for name in (*[block+'.npy' for block in BLOCKS], 'context.npz'):
            (directory / name).symlink_to((first / row['key'] / name).resolve())
    return manifest


def reference_unit(metric, selected):
    square = reference_matrix(metric, selected)
    return np.median(square[square > 1e-10])


def compare_task(output, rows, task, head_only=False):
    selected = reference_indices(output, rows)
    ref_rows = [row for row in rows if row['key'] in selected]
    sources = np.concatenate([np.repeat(row['source_id'], len(selected[row['key']])) for row in ref_rows])
    blocks = {}
    block_names = ('geometry_head', 'js', 'jacobian_head') if head_only else ('geometry_head', 'geometry_gram', 'js', 'jacobian_head', 'fisher')
    for block in block_names:
        blocks[block], _, _ = block_distances(output, rows, selected, block)
    if head_only:
        reading = blocks['geometry_head']
        adoption = {row['key']: .5*(blocks['js'][row['key']]+blocks['jacobian_head'][row['key']]) for row in rows}
    else:
        reading = {row['key']: .5*(blocks['geometry_head'][row['key']]+blocks['geometry_gram'][row['key']]) for row in rows}
        adoption = {row['key']: (blocks['js'][row['key']]+blocks['jacobian_head'][row['key']]+blocks['fisher'][row['key']])/3 for row in rows}
    raw = {row['key']: np.load(output / row['key'] / 'context.npz')['values'] for row in rows}
    fitted = fit_scale(reference_matrix(raw, selected))
    context = {key: standardize(value, fitted) for key, value in raw.items()}
    reference = reference_matrix(context, selected)
    context = {key: distances(value, reference) for key, value in context.items()}
    read_unit, adoption_unit, context_unit = [reference_unit(metric, selected) for metric in (reading, adoption, context)]
    modes = [('adoption_given_reading', adoption, reading, read_unit),
             ('reading_given_adoption', reading, adoption, adoption_unit)]
    scores = {row['key']: {} for row in rows}
    for name, metric, condition, unit in modes:
        matching = {key: .5*(context[key]/context_unit+condition[key]/unit) for key in metric}
        reference_distance = reference_matrix(metric, selected)
        reference_context = reference_matrix(matching, selected)
        for row in rows:
            key = row['key']
            query_sources = np.repeat(row['source_id'], len(metric[key]))
            value, neighbors, radius, scale = local_ratios(metric[key], reference_distance,
                matching[key], reference_context, query_sources, sources)
            scores[key][name] = value
            np.savez_compressed(output / key / (name+'_neighbors.npz'),
                indices=np.stack(neighbors), query_radius=radius, neighbor_radius=scale)
    for score in scores.values():
        score['association'] = np.sqrt(score['adoption_given_reading']*score['reading_given_adoption'])
    write_json(output / (task+'_reference.json'), dict(sources=sources.tolist(),
        identities=[(row['key'], int(token)) for row in ref_rows for token in selected[row['key']]],
        distance_units=dict(reading=float(read_unit), adoption=float(adoption_unit), context=float(context_unit))))
    return scores


def add_fusion(first, output, rows, scores):
    for row in rows:
        with np.load(first / row['key'] / 'scores.npz') as saved:
            scores[row['key']].update({name: saved[name] for name in saved.files})
    for method in ('association', 'joint_transition'):
        values, weights = [], []
        for row in rows:
            if row['role'] == 'fit':
                valid = valid_tokens(row, output, OLD)
                values.append(scores[row['key']][method][valid])
                weights.append(np.full(valid.sum(), 1/valid.sum()))
        fitted = fit_cdf(np.concatenate(values), np.concatenate(weights))
        for score in scores.values():
            rank = percentile(score[method], fitted)
            baseline = score['source_route_fixed']
            if method == 'association':
                score['association_fused'] = .75*baseline+.25*rank
                score['association_gate'] = np.sqrt(baseline*rank)
            else:
                score['transition_gate'] = np.sqrt(baseline*rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, default=Path('outputs/head_state_readout_20260928_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head-only', action='store_true')
    args = parser.parse_args()
    manifest = prepare(args.first, args.output)
    methods = NEW + tuple(read_json(args.first / 'scores_frozen.json')['methods'])
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [row for row in manifest['records'] if row['task'] == task]
        scores = compare_task(args.output, rows, task, args.head_only)
        add_fusion(args.first, args.output, rows, scores)
        for row in rows:
            np.savez_compressed(args.output / row['key'] / 'scores.npz', **scores[row['key']])
        print('association scored', task, flush=True)
    thresholds = read_json(args.first / 'thresholds.json')
    for task in thresholds:
        dev = [row for row in manifest['records'] if row['role'] == 'dev' and row['task'] == task]
        for method in NEW:
            values, weights = [], []
            for row in dev:
                valid = valid_tokens(row, args.output, OLD)
                with np.load(args.output / row['key'] / 'scores.npz') as saved:
                    values.append(saved[method][valid])
                weights.append(np.full(valid.sum(), 1/valid.sum()))
            ordered, cumulative = fit_cdf(np.concatenate(values), np.concatenate(weights))
            thresholds[task][method] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(args.output / 'thresholds.json', thresholds)
    write_json(args.output / 'scores_frozen.json', dict(methods=methods, main='association',
        labels_used=False, head_selection=False, head_only=args.head_only, status='complete',
        design_informed_by='exposed-sample errors and prior diagnostic; not independent confirmation',
        adoption='JS + per-head native gradients/FFN; head-only excludes global Fisher and Gram from scoring; raw tensors retained',
        context='half surface/generation context distance, half conditioning view distance; 16 then8 per source',
        calibration='4 dev/task source-equal unlabeled mixture95'))


if __name__ == '__main__':
    main()
