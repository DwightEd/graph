"""Run cached scalar comparisons on all three RAGTruth tasks, without label fitting.

These are the historical fixed baseline and independent token contrasts, not
the native message graph: its automatic event measurements are still missing.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from experiments.context_response.restore import cached_baselines
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.native_support.unified.calibration import fit_distribution
from experiments.probabilistic_detection.data import evaluation_labels, source_weights
from experiments.probabilistic_detection.evaluation import evaluate_method, source_bootstrap
from experiments.token_evidence.evaluate import weighted_quantile
from experiments.unsupervised_graph.fixed import fit_reference
from experiments.unsupervised_graph.scalar import rank_scores, scalar_scores
from .readout import METHODS, PRIMARY, contrasts, pack_contrasts, score_contrasts
from .span_metrics import span_metrics


PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')
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
        baseline_reference, thresholds[task]['base'] = fit_reference(pack)
        fitted[task]['fixed_baseline'] = baseline_reference
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
        scores['base'] = rank_scores(scalar_scores(pack), fitted[task]['fixed_baseline'])['fixed_unsupervised']
        np.savez_compressed(output / f'{task}_test.npz', **scores, token_id=pack['token_id'],
                            target=pack['target'], answer_index=pack['answer_index'])
        coverage[task] = dict(answers=len(metadata['records']), valid_tokens=len(pack['token_id']))
        print('test scores frozen', task, coverage[task], flush=True)
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


def run_test(output):
    fit(output)
    score_test(output)
    evaluate_test(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'pilot', 'score-test', 'evaluate-test', 'run-test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'fit': fit, 'pilot': pilot_scores, 'score-test': score_test,
     'evaluate-test': evaluate_test, 'run-test': run_test}[args.stage](args.output)


if __name__ == '__main__':
    main()
