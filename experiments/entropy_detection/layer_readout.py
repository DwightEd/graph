"""Fit source-held-out layer readouts; known cases never set model or thresholds."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import uncertainty_features
from .run import read_json, write_json, metrics
from experiments.probabilistic_detection.data import load_pack, source_weights
from experiments.probabilistic_detection.evaluation import threshold_at_fpr


def features(values, units, positions):
    values = values[positions]
    temporal = uncertainty_features(values[:, 16], values[:, 17], units, positions)
    changes = np.column_stack([values[:, base+kind]-values[:, 16+kind]
                               for base in (0, 4, 8, 12) for kind in (0, 1)])
    return dict(uncertainty=temporal, layers=np.column_stack((temporal, values, changes)))


def prepare(packs, capture, frozen):
    manifest = read_json(capture/'manifest.json')
    records = manifest['records']
    result = []
    checks = []
    for task in ('QA', 'Summary', 'Data2txt'):
        for split in ('train', 'test'):
            pack, meta = load_pack(packs, task, split)
            with np.load(frozen/task/f'{split}_scores.npz') as saved:
                frozen_joint = saved['joint']
            for record in records:
                if record['task'] != task or record['split'] != split:
                    continue
                with np.load(capture/f"{record['key']}.npz") as data:
                    values, ids = data['values'], data['token_ids']
                if record['role'] == 'manual_positive':
                    from experiments.native_support.evidence_contrast.views import unit_intervals
                    manual = record['manual_answer']
                    pieces = [manual['answer'][a:b] for a, b in manual['offsets']]
                    positions = np.flatnonzero([b > a for a, b in manual['offsets']])
                    units = np.zeros(len(ids), dtype=int)
                    for uid, unit in enumerate(unit_intervals(dict(token_text=pieces, prompt_length=0), 128)):
                        units[unit['start']:unit['stop']] = uid
                    inputs = features(values, units[positions], positions)
                    labels = np.zeros(len(positions), dtype=int)
                else:
                    region = slice(record['packed_start'], record['packed_stop'])
                    positions = pack['target'][region]
                    np.testing.assert_array_equal(ids[positions], pack['token_id'][region])
                    inputs = features(values, pack['unit_index'][region], positions)
                    static = np.column_stack((pack['context'][region], pack['observations'][region]))
                    inputs['static'] = static
                    inputs['joint'] = np.column_stack((static, inputs['layers']))
                    inputs['frozen_joint'] = frozen_joint[region]
                    labels = pack['labels'][region] if split == 'train' else None
                    checks.append(dict(key=record['key'], entropy_max_error=float(np.abs(values[positions,16]-pack['observations'][region,6]).max()),
                        surprisal_max_error=float(np.abs(values[positions,17]+pack['observations'][region,2]).max())))
                result.append(dict(record=record, positions=positions, inputs=inputs, labels=labels))
    write_json(capture/'replay_checks.json', checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.packs, args.capture, args.frozen)
    selections = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        task_rows = [row for row in rows if row['record']['task'] == task]
        groups = {role: [row for row in task_rows if row['record']['role'] == role] for role in ('fit', 'dev')}
        labels = {role: np.concatenate([row['labels'] for row in group]) for role, group in groups.items()}
        weights = source_weights(np.concatenate([np.full(len(row['labels']), index) for index, row in enumerate(groups['fit'])]))
        models, development, thresholds = {}, {}, {}
        for kind in ('uncertainty', 'layers', 'static', 'joint'):
            inputs = {role: np.concatenate([row['inputs'][kind] for row in group]) for role, group in groups.items()}
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, min_samples_leaf=40,
                l2_regularization=10., learning_rate=.05, early_stopping=False, random_state=42)
            model.fit(inputs['fit'], labels['fit'], sample_weight=weights)
            models[kind] = model
            score = model.decision_function(inputs['dev'])
            thresholds[kind] = threshold_at_fpr(labels['dev'], score)
            development[kind] = metrics(labels['dev'], score, thresholds[kind])
        selected = max(models, key=lambda kind: (development[kind]['recall'], development[kind]['ap']))
        selections[task] = dict(selected=selected, development=development, thresholds=thresholds,
            fit_tokens=len(labels['fit']), dev_tokens=len(labels['dev']),
            fit_positive=int(labels['fit'].sum()), dev_positive=int(labels['dev'].sum()))
        joblib.dump(models, args.capture/f'{task}_models.joblib')
        for row in task_rows:
            row['scores'] = {kind: model.decision_function(row['inputs'][kind]) for kind, model in models.items() if kind in row['inputs']}
            np.savez_compressed(args.capture/f"{row['record']['key']}_scores.npz", positions=row['positions'], **row['scores'])
    write_json(args.capture/'selection.json', selections)
    write_json(args.capture/'READOUT_FREEZE.json', dict(status='all case scores frozen before evaluation',
        supervision=True, no_known_source_in_fit_dev=True))
    print(selections, flush=True)


if __name__ == '__main__':
    main()
