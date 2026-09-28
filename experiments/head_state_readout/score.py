"""Freeze full-head scores and mixture thresholds before annotation access."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from .density import REFERENCE_PER_SOURCE, distances, fit_scale, standardize, local_ratios
from .features import BLOCKS

METHODS = ('joint_conditional', 'joint_unconditional', 'joint_transition', 'geometry', 'js',
           'jacobian', 'state_fused', 'transition_fused', 'source_route_fixed',
           'raw_route_offline_mean', 'net_route_offline_mean')
OLD = Path('outputs/transport_topology_cases_20260928')


def reference_indices(output, rows):
    indices = {}
    for row in rows:
        if row['role'] == 'fit':
            available = np.flatnonzero(valid_tokens(row, output, OLD))
            chosen = np.linspace(0, len(available)-1, min(REFERENCE_PER_SOURCE, len(available))).astype(int)
            indices[row['key']] = available[chosen]
    return indices


def differences(values):
    return np.diff(values, axis=0, prepend=values[:1])


def block_distances(output, rows, selected, block):
    raw = {row['key']: np.load(output / row['key'] / (block + '.npy'), mmap_mode='r') for row in rows}
    reference = np.concatenate([raw[key][indices] for key, indices in selected.items()])
    fitted = fit_scale(reference)
    assert np.isfinite(fitted[0]).all(), block
    scaled = {key: standardize(value, fitted) for key, value in raw.items()}
    reference = np.concatenate([scaled[key][indices] for key, indices in selected.items()])
    reference_delta = np.concatenate([differences(scaled[key])[indices] for key, indices in selected.items()])
    current = {key: distances(value, reference) for key, value in scaled.items()}
    transition = {key: distances(differences(value), reference_delta) for key, value in scaled.items()}
    return current, transition, fitted


def combine_blocks(blocks):
    geometry = .5 * (blocks['geometry_head'] + blocks['geometry_gram'])
    jacobian = (blocks['jacobian_head'] + blocks['tangent'] + blocks['fisher']) / 3
    return dict(geometry=geometry, js=blocks['js'], jacobian=jacobian,
                joint=(geometry + blocks['js'] + jacobian) / 3)


def reference_matrix(values, selected):
    return np.concatenate([values[key][indices] for key, indices in selected.items()])


def score_task(output, rows, task):
    selected = reference_indices(output, rows)
    ref_rows = [row for row in rows if row['key'] in selected]
    ref_sources = np.concatenate([np.repeat(row['source_id'], len(selected[row['key']])) for row in ref_rows])
    ref_identity = [(row['key'], int(index)) for row in ref_rows for index in selected[row['key']]]
    distances_by_block, delta_by_block, scalers = {}, {}, {}
    for block in BLOCKS:
        current, transition, fitted = block_distances(output, rows, selected, block)
        distances_by_block[block] = current
        delta_by_block[block] = transition
        scalers[block + '_center'], scalers[block + '_scale'] = fitted
        print(task, 'distances', block, flush=True)
    raw_context = {row['key']: np.load(output / row['key'] / 'context.npz')['values'] for row in rows}
    context_fit = fit_scale(reference_matrix(raw_context, selected))
    context = {key: standardize(value, context_fit) for key, value in raw_context.items()}
    ref_context = reference_matrix(context, selected)
    context_distance = {key: distances(value, ref_context) for key, value in context.items()}
    current = {row['key']: combine_blocks({block: distances_by_block[block][row['key']] for block in BLOCKS}) for row in rows}
    delta = {row['key']: combine_blocks({block: delta_by_block[block][row['key']] for block in BLOCKS}) for row in rows}
    specifications = dict(geometry=('geometry', True, False), js=('js', True, False),
        jacobian=('jacobian', True, False), joint_conditional=('joint', True, False),
        joint_unconditional=('joint', False, False), joint_transition=('joint', True, True))
    scores = {row['key']: {} for row in rows}
    for method, (view, conditional, use_delta) in specifications.items():
        metric = {key: .5*(value[view]+delta[key][view]) if use_delta else value[view] for key, value in current.items()}
        ref_distance = reference_matrix(metric, selected)
        ref_context_distance = reference_matrix(context_distance, selected)
        for row in rows:
            key = row['key']
            query_sources = np.repeat(row['source_id'], len(metric[key]))
            score, neighbors, radius, scale = local_ratios(metric[key], ref_distance,
                context_distance[key], ref_context_distance, query_sources, ref_sources, conditional)
            scores[key][method] = score
            np.savez_compressed(output / key / (method + '_neighbors.npz'),
                indices=np.stack(neighbors), query_radius=radius, neighbor_radius=scale)
    np.savez_compressed(output / (task+'_scales.npz'), **scalers,
                        context_center=context_fit[0], context_scale=context_fit[1])
    write_json(output / (task+'_reference.json'), dict(identities=ref_identity,
        sources=ref_sources.tolist(), labels_used=False, per_source_limit=REFERENCE_PER_SOURCE))
    return scores


def add_baselines(output, rows, scores, base):
    for row in rows:
        with np.load(base / row['key'] / 'scores.npz') as previous:
            for name in METHODS[-3:]:
                scores[row['key']][name] = previous[name].copy()
    for new, method in [('state_fused', 'joint_conditional'), ('transition_fused', 'joint_transition')]:
        values, weights = [], []
        for row in rows:
            if row['role'] == 'fit':
                valid = valid_tokens(row, output, OLD)
                values.append(scores[row['key']][method][valid])
                weights.append(np.full(valid.sum(), 1/valid.sum()))
        reference = fit_cdf(np.concatenate(values), np.concatenate(weights))
        for row in rows:
            score = scores[row['key']]
            score[new] = .75*score['source_route_fixed'] + .25*percentile(score[method], reference)


def calibrate(output, records):
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        values = {name: [] for name in METHODS}
        weights = []
        for row in records:
            if row['role'] != 'dev' or row['task'] != task:
                continue
            valid = valid_tokens(row, output, OLD)
            weights.append(np.full(valid.sum(), 1/valid.sum()))
            with np.load(output / row['key'] / 'scores.npz') as saved:
                for name in METHODS:
                    values[name].append(saved[name][valid])
        thresholds[task] = {}
        for name in METHODS:
            ordered, cumulative = fit_cdf(np.concatenate(values[name]), np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(output / 'thresholds.json', thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'features_complete.json')
    manifest = read_json(args.output / 'manifest.json')
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [row for row in manifest['records'] if row['task'] == task]
        scores = score_task(args.output, rows, task)
        add_baselines(args.output, rows, scores, Path(manifest['route_base']))
        for row in rows:
            np.savez_compressed(args.output / row['key'] / 'scores.npz', **scores[row['key']])
    calibrate(args.output, manifest['records'])
    write_json(args.output / 'scores_frozen.json', dict(methods=METHODS, main='joint_conditional',
        labels_used=False, head_selection=False, reference='4 fit sources/task; max64 evenly spaced tokens/source',
        scale='fit median/IQR per physical coordinate; undefined features median-imputed plus missingness flags',
        distance='all coordinates retained; geometry/JS/Jacobian views weighted equally',
        neighbors='16 context neighbors/source then8 state neighbors/source; same source always excluded',
        threshold='4 dev/task source-equal unlabeled mixture95', status='complete'))


if __name__ == '__main__':
    main()
