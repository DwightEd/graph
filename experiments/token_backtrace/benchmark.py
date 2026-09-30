"""Fit without labels, diagnose exposed cases, then freeze all 2700 test answers."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from experiments.context_response.restore import cached_baselines, original_threshold
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.native_support.unified.calibration import fit_distribution
from experiments.probabilistic_detection.data import evaluation_labels, source_weights
from experiments.probabilistic_detection.evaluation import evaluate_method, source_bootstrap
from experiments.token_evidence.evaluate import weighted_quantile
from .readout import METHODS, PRIMARY, contrasts, pack_contrasts, score_contrasts
from .span_metrics import span_metrics


PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')
BASELINE = Path('outputs/unsupervised_graph_20260928')
TASKS = ('QA', 'Summary', 'Data2txt')
PACK_KEYS = ('context', 'observations', 'token_id', 'target', 'source_index',
             'answer_index', 'unit_index', 'development')
CASES = ('15604', '219', '11907', '12015', '12045', '12219', '9022', '7305')


def load_pack(task, split):
    metadata = read_json(PACKS / f'{task}_{split}.json')
    with np.load(PACKS / f'{task}_{split}.npz') as arrays:
        pack = {name: arrays[name] for name in PACK_KEYS}
    return pack, metadata


def fit(output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', dict(primary=PRIMARY, methods=METHODS,
        formula='0.75 source-contrast midrank + 0.25 raw-route midrank',
        calibration='official-train fit CDF; source-balanced unlabeled dev mixture95',
        probability_resolution=float(np.finfo(np.float32).eps),
        labels_used_for_fit_or_threshold=False, labels_seen_in_mechanism_design=True,
        test_previously_exposed=True, future_risk_averaging=False,
        local_condition='saved variable unit history reset, ablation only; never risk averaging',
        primary_condition='full original response history',
        source_removal='historical prompt deletion changes positions; not isolated edge intervention',
        model='Llama-3.1-8B-Instruct observer, not necessarily original generator',
        pilot_cases=CASES, created_utc=datetime.now(timezone.utc).isoformat()))
    fitted, thresholds = {}, {}
    for task in TASKS:
        pack, metadata = load_pack(task, 'train')
        raw = pack_contrasts(pack, metadata)
        development = pack['development']
        fitted[task] = {name: fit_distribution(value[~development], pack['source_index'][~development])
                        for name, value in raw.items()}
        scores = score_contrasts(raw, fitted[task])
        weights = source_weights(pack['source_index'][development])
        thresholds[task] = {name: weighted_quantile(value[development], weights, .95)
                            for name, value in scores.items()}
        thresholds[task]['base'] = original_threshold(task)
        print('fitted without labels', task, len(pack['token_id']), flush=True)
    joblib.dump(fitted, output / 'calibration.joblib')
    write_json(output / 'thresholds.json', thresholds)
    write_json(output / 'fit_complete.json', dict(complete=True, labels_accessed=False))


def pilot_scores(output):
    fitted = joblib.load(output / 'calibration.joblib')
    rows = read_json('outputs/token_evidence_20260929_v1/manifest.json')['records']
    rows = [row for row in rows if row['key'] in CASES]
    baseline = cached_baselines(set(CASES))
    for row in rows:
        original = row['original']
        directory = Path(original['root']) / original['directory']
        with np.load(directory / 'with_source.npz') as present, np.load(directory / 'without_source.npz') as absent:
            np.testing.assert_array_equal(present['token_id'], row['response']['answer_ids'])
            np.testing.assert_array_equal(absent['token_id'], present['token_id'])
            raw = contrasts(present['full'].astype(float), absent['full'].astype(float),
                            present['local'].astype(float), absent['local'].astype(float), present['raw_route'])
        scores = score_contrasts(raw, fitted[row['task']])
        destination = output / 'pilot' / row['key']
        destination.mkdir(parents=True)
        np.savez_compressed(destination / 'scores.npz', **scores, base=baseline[row['key']],
                            token_id=row['response']['answer_ids'], **{f'raw_{k}': v for k, v in raw.items()})
    write_json(output / 'pilot_frozen.json', dict(records=rows, labels_accessed=False))


def score_test(output):
    fitted = joblib.load(output / 'calibration.joblib')
    coverage = {}
    for task in TASKS:
        pack, metadata = load_pack(task, 'test')
        scores = score_contrasts(pack_contrasts(pack, metadata), fitted[task])
        with np.load(BASELINE / task / 'test_scores.npz') as baseline:
            scores['base'] = baseline['fixed_unsupervised']
        assert all(len(value) == len(pack['token_id']) and np.isfinite(value).all() for value in scores.values())
        np.savez_compressed(output / f'{task}_test.npz', **scores, token_id=pack['token_id'],
                            target=pack['target'], answer_index=pack['answer_index'])
        coverage[task] = dict(answers=len(metadata['records']), valid_tokens=len(pack['token_id']))
        print('test scores frozen', task, coverage[task], flush=True)
    assert sum(row['answers'] for row in coverage.values()) == 2700
    assert sum(row['valid_tokens'] for row in coverage.values()) == 424408
    write_json(output / 'test_frozen.json', dict(coverage=coverage, labels_accessed=False,
        new_llm_forwards=0, measurement='reuse all original native scalar caches',
        root_gradients_recomputed_on_full_test=False, frozen_utc=datetime.now(timezone.utc).isoformat()))


def evaluate_test(output, methods=METHODS, primary=PRIMARY):
    read_json(output / 'test_frozen.json')
    thresholds = read_json(output / 'thresholds.json')
    results = {}
    for task in TASKS:
        pack, metadata = load_pack(task, 'test')
        pack.update(evaluation_labels(Path(metadata['source_cache']), pack, metadata))
        with np.load(output / f'{task}_test.npz') as saved:
            np.testing.assert_array_equal(pack['token_id'], saved['token_id'])
            scores = {name: saved[name] for name in (*methods, 'base')}
        results[task] = {}
        for name, value in scores.items():
            result = evaluate_method(pack, value, thresholds[task][name])
            alarm = value > thresholds[task][name]
            result.update(tp=int(np.sum(alarm & (pack['labels'] == 1))),
                          fp=int(np.sum(alarm & (pack['labels'] == 0))),
                          positives=int(pack['labels'].sum()),
                          threshold_rule='frozen unlabeled dev mixture95; base historical threshold')
            if name == 'logic_only':
                result['threshold_rule'] = 'fixed 0.5 for binary contradictions; unknown makes no alarm'
            elif name == 'logic_full':
                result['threshold_rule'] = 'inherit native odds_full threshold; no mixture95 guarantee after logic overrides'
            results[task][name] = result
        results[task]['primary_vs_base'] = source_bootstrap(pack, scores[primary], scores['base'])
        character = span_metrics(Path(metadata['source_cache']), metadata, pack, scores, thresholds[task])
        for name in scores:
            results[task][name]['characters'] = character[name]
        np.savez_compressed(output / f'{task}_evaluation_labels.npz',
                            **{k: pack[k] for k in ('labels', 'onsets', 'firsts')})
        print(task, {k: (v['auroc'], v['tp'], v['fp']) for k, v in results[task].items() if k != 'primary_vs_base'}, flush=True)
    write_json(output / 'test_results.json', results)


def load_graph_records(preserve_mass=False):
    from .diagnose import TRACES
    from .readout import graph_designs

    rows = read_json('outputs/token_evidence_20260929_v1/manifest.json')['records']
    records, inputs = [], []
    for row in (row for row in rows if row['key'] in CASES):
        trace_path = next(root / row['key'] / 'trace.npz' for root in TRACES if (root / row['key']).exists())
        state_path = Path('outputs/automatic_evidence_20260930_v2') / row['key'] / 'effects.npz'
        with np.load(trace_path) as trace, np.load(state_path) as states:
            np.testing.assert_array_equal(trace['token_ids'], row['response']['answer_ids'])
            np.testing.assert_array_equal(states['token_ids'], trace['token_ids'])
            prompt = len(row['prompt'])
            for target, effect in enumerate(trace['root_effect']):
                assert not np.any(effect[prompt + target:]), 'future response root'
            attributes, designs, weights = graph_designs(row, trace, states['original_hidden'], preserve_mass)
            source_effect = trace['root_effect'][:, :prompt].copy()
        offsets = np.asarray(row['response']['offsets'])
        valid = (offsets[:, 1] > offsets[:, 0]) & ~np.isin(
            row['response']['answer_ids'], row['response']['special_ids'])
        records.append(dict(key=row['key'], source_id=row['original']['source_id'], row=row,
            attributes=attributes, designs=designs, weights=weights, valid=valid,
            source_effect=source_effect))
        inputs.extend((trace_path, state_path))
    assert len(records) == 8 and sum(len(r['valid']) for r in records) == 1487
    return records, inputs


def score_graph(output, preserve_mass=False):
    import hashlib
    from time import perf_counter
    from .readout import GRAPH_METHODS, MASS_METHODS
    from .readout import crossfit_graph

    prefix = 'mass_' if preserve_mass else ''
    methods = MASS_METHODS if preserve_mass else GRAPH_METHODS
    if preserve_mass:
        read_json(output / 'scores_frozen.json')
        assert not (output / 'mass_protocol.json').exists(), 'refuse to overwrite mass experiment'
    else:
        output.mkdir(parents=True, exist_ok=False)
    protocol = dict(primary='graph_ridge', methods=[*GRAPH_METHODS, 'base'], seeds=[17, 29, 43],
        fit='outer source exclusion, inner held-source calibration; source-balanced scaling/ridge/IF',
        threshold=.95, high_score_is_anomalous=True, labels_used=False,
        hidden='cached final normalized 4096-state projected to 32 dimensions, fixed seed17',
        graph='signed history root-gradient total-response edges, one-hop contexts; no path rollout',
        source='full prompt root responses saved; source signed mass attributes, not semantic evidence',
        null='within log2-lag and repeated-current-token strata; signed row multiset preserved',
        score='equal mean squared standardized reconstruction error of hidden and six scalar blocks',
        ridge_alpha=1., forest_trees=100, forest_max_samples=256,
        natural_cases_previously_exposed=True, independent_test=False,
        model='Llama-3.1-8B-Instruct observer replay', new_llm_forwards=0,
        risk_averaging=False, cross_answer_edges=False,
        projection='fixed Gaussian; no claim of exact distance preservation',
        calibration_caveat='inner models fit six sources, outer fit seven; percentile not calibrated truth probability')
    if preserve_mass:
        protocol.update(primary='graph_mass', methods=[*GRAPH_METHODS, *MASS_METHODS, 'base'],
            posthoc_after_first_evaluation=True,
            context='raw signed channel @ attributes, without row normalization')
    write_json(output / (prefix + 'protocol.json'), protocol)
    started = perf_counter()
    records, inputs = load_graph_records(preserve_mass)
    code = [Path(__file__), Path(__file__).with_name('readout.py')]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in (*inputs, *code)}
    write_json(output / (prefix + 'inputs.json'), hashes)
    baseline = cached_baselines({row['key'] for row in records})
    scores = {row['key']: dict(base=baseline[row['key']]) for row in records}
    if preserve_mass:
        for key in scores:
            with np.load(output / key / 'scores.npz') as original:
                scores[key] = {name: original[name] for name in original.files
                               if name not in ('valid', 'token_ids')}
    folds = {}
    for method in methods:
        predictions, folds[method] = crossfit_graph(records, method)
        for key, values in predictions.items():
            scores[key][method] = values['score']
            scores[key]['raw_' + method] = values['raw']
        print('graph scores', method, round(perf_counter()-started, 2), flush=True)
    write_json(output / (prefix + 'calibration.json'), folds)
    for record in records:
        key = record['key']
        directory = output / key
        directory.mkdir(exist_ok=preserve_mass)
        for values in scores[key].values():
            assert np.isfinite(values[record['valid']]).all()
        np.savez_compressed(directory / (prefix + 'scores.npz'), **scores[key],
            token_ids=record['row']['response']['answer_ids'], valid=record['valid'])
        np.savez_compressed(directory / (prefix + 'graph.npz'), attributes=record['attributes'],
            history_effect=record['weights'], prompt_effect=record['source_effect'],
            **record['designs'])
    if not preserve_mass:
        write_json(output / 'manifest.json', dict(records=[record['row'] for record in records]))
    write_json(output / (prefix + 'scores_frozen.json'), dict(status='complete', labels_accessed=False,
        answers=len(records), tokens=sum(len(r['valid']) for r in records),
        seconds=perf_counter()-started, methods=protocol['methods']))
    print('FROZEN', round(perf_counter()-started, 2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'pilot', 'score-test', 'evaluate-test',
        'graph-score', 'graph-evaluate', 'graph-mass-score', 'graph-mass-evaluate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ('graph-evaluate', 'graph-mass-evaluate'):
        from .diagnose import evaluate_graph
        evaluate_graph(args.output, preserve_mass=args.stage == 'graph-mass-evaluate')
        return
    if args.stage == 'graph-mass-score':
        score_graph(args.output, preserve_mass=True)
        return
    {'fit': fit, 'pilot': pilot_scores, 'score-test': score_test,
     'evaluate-test': evaluate_test, 'graph-score': score_graph}[args.stage](args.output)


if __name__ == '__main__':
    main()
