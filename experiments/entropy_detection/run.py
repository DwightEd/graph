"""Fit uncertainty readouts excluding exposed sources; evaluate frozen cases/test."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'teaching/state_audit/src'))
from experiments.probabilistic_detection.data import load_pack, source_weights, evaluation_labels
from experiments.probabilistic_detection.cases import KNOWN_CASES, build_case, render_html
from experiments.probabilistic_detection.evaluation import threshold_at_fpr, evaluate_method
from experiments.native_support.ragtruth_benchmark.data import annotations
from .features import matrices, feature_names


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def metrics(labels, values, threshold):
    alarm = values > threshold
    return dict(auroc=float(roc_auc_score(labels, values)),
        ap=float(average_precision_score(labels, values)), threshold=float(threshold),
        recall=float(alarm[labels == 1].mean()), fpr=float(alarm[labels == 0].mean()))


def exposed_sources(packs):
    return {r['source_id'] for task in ('QA', 'Summary', 'Data2txt') for split in ('train', 'test')
        for r in read_json(packs/f'{task}_{split}.json')['records'] if r['id'] in KNOWN_CASES}


def allowed_mask(metadata, excluded, count):
    allowed = np.ones(count, dtype=bool)
    for record in metadata['records']:
        if record['source_id'] in excluded:
            allowed[record['packed_start']:record['packed_stop']] = False
    return allowed


def basic_scores(pack, matrix):
    return dict(source_first=pack['baselines'][:, 0], source_refine=pack['baselines'][:, 1],
        entropy=matrix[:, 0], entropy_peak4=matrix[:, 6], entropy_peak8=matrix[:, 7],
        entropy_future4=matrix[:, 8], surprisal=matrix[:, 1])


def fit_task(packs, output, task, excluded, maximum_tokens):
    pack, metadata = load_pack(packs, task, 'train')
    inputs = matrices(pack, metadata['records'])
    allowed = allowed_mask(metadata, excluded, len(pack['target']))
    train = np.flatnonzero(allowed & ~pack['development'])
    development = np.flatnonzero(allowed & pack['development'])
    rng = np.random.default_rng(42)
    if len(train) > maximum_tokens:
        train = np.sort(rng.choice(train, maximum_tokens, replace=False))
    train_sources = set(pack['source_index'][train].tolist())
    assert train_sources.isdisjoint(pack['source_index'][development].tolist())
    weights = source_weights(pack['source_index'][train])
    directory = output/task
    directory.mkdir(parents=True)
    scores = basic_scores(pack, inputs['uncertainty'])
    models = {}
    for kind in ('uncertainty', 'static', 'joint'):
        model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=5., learning_rate=.08,
            early_stopping=False, random_state=42)
        model.fit(inputs[kind][train], pack['labels'][train], sample_weight=weights)
        models[kind] = model
        scores[kind] = model.decision_function(inputs[kind])
        print(task, kind, 'fitted', flush=True)
    thresholds = {name: threshold_at_fpr(pack['labels'][development], values[development])
                  for name, values in scores.items()}
    development_metrics = {name: metrics(pack['labels'][development], values[development], thresholds[name])
                           for name, values in scores.items()}
    selected = max(('uncertainty', 'static', 'joint'), key=lambda name: (
        development_metrics[name]['recall'], development_metrics[name]['ap']))
    selection = dict(selected=selected, thresholds=thresholds,
        rule='highest development token recall at own 5% normal FPR, AP tie break',
        development=development_metrics, train_tokens=len(train), development_tokens=len(development),
        excluded_sources=sorted(excluded), labels_used_for_fit=True,
        features={kind: feature_names(kind) for kind in models}, future_observation=True)
    joblib.dump(models, directory/'models.joblib')
    write_json(directory/'selection.json', selection)
    np.savez_compressed(directory/'train_scores.npz', **scores)
    return selection


def score_test(packs, output, task):
    pack, metadata = load_pack(packs, task, 'test')
    inputs = matrices(pack, metadata['records'])
    models = joblib.load(output/task/'models.joblib')
    scores = basic_scores(pack, inputs['uncertainty'])
    scores.update({kind: model.decision_function(inputs[kind]) for kind, model in models.items()})
    np.savez_compressed(output/task/'test_scores.npz', **scores)
    write_json(output/task/'test_frozen.json', dict(status='frozen', tokens=len(pack['target'])))


def evaluate_cases(packs, output, task):
    cases = []
    selection = read_json(output/task/'selection.json')
    for split in ('train', 'test'):
        pack, metadata = load_pack(packs, task, split)
        root = Path(metadata['source_cache'])
        chosen = [r for r in metadata['records'] if r['id'] in KNOWN_CASES]
        truth = annotations(root, read_json(root/'manifest.json'), chosen)
        with np.load(output/task/f'{split}_scores.npz') as saved:
            scores = {name: saved[name] for name in saved.files}
        for record in chosen:
            region = slice(record['packed_start'], record['packed_stop'])
            response = read_json(root/record['directory']/'response.json')
            available = {name: values[region] for name, values in scores.items()
                         if np.isfinite(values[region]).all()}
            case = build_case(record, response, truth[record['id']], pack['target'][region],
                pack['token_id'][region], available,
                selection['thresholds'], 8)
            case['partition_note'] = '已知暴露回归；本轮整来源排除拟合/阈值校准，不能作为未见泛化。'
            case['new_fit_source_excluded'] = True
            case['in_sample'] = False
            case['methods_unavailable'] = sorted(set(scores) - set(available))
            case['selected'] = selection['selected']
            cases.append(case)
    return cases


def evaluate_test(packs, output, task):
    pack, metadata = load_pack(packs, task, 'test')
    truth = evaluation_labels(Path(metadata['source_cache']), pack, metadata)
    selection = read_json(output/task/'selection.json')
    with np.load(output/task/'test_scores.npz') as saved:
        scores = {name: saved[name] for name in saved.files}
    result = {name: evaluate_method(dict(pack, **truth), values, selection['thresholds'][name])
              for name, values in scores.items()}
    write_json(output/task/'test_metrics.json', result)
    return result


def evaluate_frozen(packs, output):
    if not (output/'SCORING_FREEZE.json').exists():
        raise ValueError('Freeze all tasks before evaluation')
    cases = []
    summary = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        cases.extend(evaluate_cases(packs, output, task))
        summary[task] = evaluate_test(packs, output, task)
    write_json(output/'cases.json', dict(cases=cases))
    (output/'cases.html').write_text(render_html(dict(cases=cases)), encoding='utf-8')
    write_json(output/'summary.json', summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-train-tokens', type=int, default=150000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new experiment output directory')
    args.output.mkdir(parents=True)
    excluded = exposed_sources(args.packs)
    write_json(args.output/'protocol.json', dict(packs=str(args.packs),
        excluded_sources=sorted(excluded), max_train_tokens=args.max_train_tokens,
        training='light supervision, source-disjoint fit/dev; known sources excluded',
        fixed_scores='entropy and entropy peaks use no labels; thresholds use dev negatives',
        primary_goal='known-case recall with explicit normal false alarms', full_test='exploratory'))
    for task in ('QA', 'Summary', 'Data2txt'):
        fit_task(args.packs, args.output, task, excluded, args.max_train_tokens)
        score_test(args.packs, args.output, task)
    write_json(args.output/'SCORING_FREEZE.json', dict(status='all_task_predictions_frozen'))
    evaluate_frozen(args.packs, args.output)
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
