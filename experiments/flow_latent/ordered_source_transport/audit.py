"""Independent NumPy score/metric readback and paired source bootstrap."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..provenance_joint_state.matrix_audit import sha256, write_json
from ..provenance_joint_state.source_control_coverage import passage_records


def verify_controls(controls):
    index = {row['id']: row for row in controls}
    fit = {row['source_id'] for row in controls if row['cohort'] == 'fit'}
    dev = {row['source_id'] for row in controls if row['cohort'] == 'dev'}
    assert not fit & dev
    normalized_query_rows = 0
    for row in controls:
        begin, end = row['source_char_span']
        if row['family'].startswith('qa_'):
            passages = dict(passage_records(row['prompt'][begin:end]))
            query = row['prompt'].split('\nExact quotation:\n', 1)[1].split('<|eot_id|>', 1)[0]
            assert query == row['quote'].rstrip()
            normalized_query_rows += int(query != row['quote'])
            owners = [i for i in (2, 3) if query in passages[i]]
            correct = owners[0] - 2 if row['family'] == 'qa_owner' else int(query not in passages[3])
            assert len(owners) == 1 and row['correct'] == correct
        else:
            value = json.loads(row['prompt'][begin:end])
            for key in row['path']:
                value = value[key]
            assert row['correct'] == int(not value)
        world = 'swapped' if row['world'] == 'original' else 'original'
        partner = index[row['id'].replace('/' + row['world'] + '/', '/' + world + '/')]
        assert partner['correct'] == 1 - row['correct']
        assert partner['candidate_ids'] == row['candidate_ids']
        assert partner['prompt'][partner['source_char_span'][1]:] == row['prompt'][end:]
    return dict(controls=len(controls), fit_sources=len(fit), dev_sources=len(dev),
                trailing_whitespace_normalized_query_rows=normalized_query_rows,
                source_truth_and_swap='PASS', disjoint_sources='PASS')


def numpy_score(state, candidates, checkpoint, permutation):
    mean = checkpoint['mean'].numpy()
    scale = checkpoint['scale'].numpy()
    values = (state - mean) / scale
    variant = checkpoint['variant']
    if variant == 'single':
        values = np.broadcast_to(values[-1:], values.shape)
    elif variant == 'mean':
        values = np.broadcast_to(values.mean(axis=0, keepdims=True), values.shape)
    elif variant == 'shuffled':
        values = values[permutation]
    elif variant == 'drop_source':
        values = values.copy()
        values[:, :, 3] = 0
    weights = {name: tensor.numpy() for name, tensor in checkpoint['state_dict'].items()}
    if variant == 'wide_single':
        level = np.einsum('lsd,ls->d', values[-1], weights['level']) * weights['coordinate_linear']
        square = np.einsum('lsd,ls->d', values[-1] ** 2, weights['square']) * weights['coordinate_square']
        return candidates @ (level + square) / np.sqrt(values.shape[-1]) + weights['bias']
    level = np.einsum('tlsd,tls->d', values, weights['level'])
    transition = np.einsum('tlsd,tls->d', values[:-1] * values[1:], weights['transition'])
    return candidates @ ((level + transition) * weights['coordinate']) / np.sqrt(values.shape[-1]) + weights['bias']


def check_run(directory, states, candidates, controls, observations, permutations):
    frozen = json.loads((directory / 'FREEZE.json').read_text())
    assert frozen['scores_sha256'] == sha256(directory / 'scores.npy')
    assert frozen['model_sha256'] == sha256(directory / 'model.pt')
    scores = np.load(directory / 'scores.npy')
    checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
    selected = [0, len(controls) // 2, len(controls) - 1]
    error = max(float(np.max(np.abs(numpy_score(np.array(states[i]), np.array(candidates[i]),
                                               checkpoint, permutations[i]) - scores[i]))) for i in selected)
    assert error < 1e-3, 'Independent full-coordinate score differs'
    saved = json.loads((directory / 'RESULTS.json').read_text())['metrics']
    correct = np.array([row['correct'] for row in controls])
    native = np.array([np.argmax(row['native']['pair_logits']) for row in observations])
    covered = np.array([row['native']['greedy_token'] in control['candidate_ids']
                        for row, control in zip(observations, controls)])
    confidence = np.array([row['native']['greedy_probability'] for row in observations])
    predicted = scores.argmax(axis=1)
    risk = scores[np.arange(len(scores)), 1 - native] - scores[np.arange(len(scores)), native]
    for cohort in ('fit', 'dev'):
        for family in ['all'] + sorted({row['family'] for row in controls}):
            chosen = np.array([row['cohort'] == cohort and
                               (family == 'all' or row['family'] == family) for row in controls])
            wrong, right = chosen & (native != correct), chosen & (native == correct)
            row = saved[f'{cohort}/{family}']
            assert row['count'] == int(chosen.sum())
            assert row['pair_accuracy'] == float((predicted[chosen] == correct[chosen]).mean())
            assert row['corrected_native_wrong'] == int((predicted[wrong] == correct[wrong]).sum())
            assert row['damaged_native_right'] == int((predicted[right] != correct[right]).sum())
            assert row['native_accuracy'] == float((native[chosen] == correct[chosen]).mean())
            assert row['native_wrong'] == int(wrong.sum())
            assert row['greedy_candidate_coverage'] == int((chosen & covered).sum())
            assert row['confident_native_wrong'] == int((wrong & covered & (confidence >= .5)).sum())
            margin = np.where(correct[chosen] == 0, 1, -1) * (scores[chosen, 0] - scores[chosen, 1])
            assert abs(row['mean_signed_margin'] - float(margin.mean())) < 1e-12
            errors = (native != correct)[chosen]
            risk_defined = bool(errors.any() and (~errors).any())
            if 'program_token_risk_defined' in row:
                assert row['program_token_risk_defined'] == risk_defined
            if not risk_defined:
                assert row['program_token_risk_auroc'] is None and row['program_token_risk_ap'] is None
            if row.get('program_token_risk_auroc') is not None:
                auc, ap = independent_metrics((native != correct)[chosen], risk[chosen])
                assert abs(auc - row['program_token_risk_auroc']) < 1e-12
                assert abs(ap - row['program_token_risk_ap']) < 1e-12
    coverage = covered_choice_metrics(scores, controls, observations)
    return dict(run=directory.name, score_rows_checked=selected, independent_max_abs_error=error,
                all_group_metrics='PASS', covered_greedy_choice_metrics=coverage), (predicted == correct).astype(float)


def covered_choice_metrics(scores, controls, observations):
    """Native error/correction claims apply only where the actual greedy token is covered."""
    covered = np.array([row['native']['greedy_token'] in control['candidate_ids']
                        for row, control in zip(observations, controls)])
    native = np.array([np.argmax(row['native']['pair_logits']) for row in observations])
    truth = np.array([row['correct'] for row in controls])
    predicted = scores.argmax(axis=1)
    risk = scores[np.arange(len(scores)), 1 - native] - scores[np.arange(len(scores)), native]
    output = {}
    for cohort in ('fit', 'dev'):
        for family in ['all'] + sorted({row['family'] for row in controls}):
            chosen = covered & np.array([row['cohort'] == cohort and
                (family == 'all' or row['family'] == family) for row in controls])
            wrong, right = chosen & (native != truth), chosen & (native == truth)
            errors = (native != truth)[chosen]
            defined = bool(errors.any() and (~errors).any())
            auc, ap = independent_metrics(errors, risk[chosen]) if defined else (None, None)
            output[f'{cohort}/{family}'] = dict(count=int(chosen.sum()), native_wrong=int(wrong.sum()),
                corrected_native_wrong=int((predicted[wrong] == truth[wrong]).sum()),
                damaged_native_right=int((predicted[right] != truth[right]).sum()),
                program_token_risk_auroc=auc, program_token_risk_ap=ap)
    return output


def independent_metrics(truth, risk):
    positives = risk[truth]
    negatives = risk[~truth]
    differences = positives[:, None] - negatives[None]
    auc = float(((differences > 0) + .5 * (differences == 0)).mean())
    ap, old_recall = 0., 0.
    for threshold in np.unique(risk)[::-1]:
        chosen = risk >= threshold
        true_positive = int((chosen & truth).sum())
        recall = true_positive / int(truth.sum())
        ap += (recall - old_recall) * true_positive / int(chosen.sum())
        old_recall = recall
    return auc, float(ap)


def bootstrap_comparisons(correctness, controls):
    dev_sources = sorted({row['source_id'] for row in controls if row['cohort'] == 'dev'})
    grouped = {}
    for variant, arrays in correctness.items():
        mean_over_seeds = np.mean(arrays, axis=0)
        grouped[variant] = np.array([mean_over_seeds[[row['source_id'] == source for row in controls]].mean()
                                    for source in dev_sources])
    generator = np.random.default_rng(2026)
    draws = generator.integers(len(dev_sources), size=(2000, len(dev_sources)))
    output = {}
    for other in sorted(set(grouped) - {'ordered'}):
        difference = grouped['ordered'] - grouped[other]
        means = difference[draws].mean(axis=1)
        output['ordered-minus-' + other] = dict(point=float(difference.mean()),
            source_ci95=np.quantile(means, [.025, .975]).tolist(),
            per_source=difference.tolist(), independent_source_count=len(dev_sources))
    return dict(scope='development; paired source resampling of accuracy averaged over seeds',
                dev_sources=dev_sources, comparisons=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--fits', type=Path, required=True)
    args = parser.parse_args()
    controls = json.loads((args.cache / 'controls.json').read_text())
    observations = json.loads((args.cache / 'observations.json').read_text())
    contract = verify_controls(controls)
    states = np.load(args.cache / 'states.npy', mmap_mode='r')
    candidates = np.load(args.cache / 'candidates.npy', mmap_mode='r')
    statistics = torch.load(args.fits / 'statistics.pt', map_location='cpu', weights_only=False)
    permutations = statistics['permutations'].numpy()
    results, correctness = [], {}
    for directory in sorted(args.fits.glob('*_seed*')):
        report, values = check_run(directory, states, candidates, controls, observations, permutations)
        results.append(report)
        variant = directory.name.rsplit('_seed', 1)[0]
        correctness.setdefault(variant, []).append(values)
    comparison = bootstrap_comparisons(correctness, controls)
    write_json(args.fits / 'SOURCE_BOOTSTRAP.json', comparison)
    paths = list(args.cache.glob('*.npy')) + list(args.cache.glob('*.json'))
    paths += [args.fits / 'PROTOCOL.json', args.fits / 'statistics.pt', Path(__file__)]
    report = dict(status='PASS', scope='same-agent separate NumPy arithmetic, not independent scientific acceptance',
        new_llm_forwards=0, new_fits=0, contracts=contract, runs=results,
        hashes={str(path): sha256(path) for path in paths})
    write_json(args.fits / 'NUMERIC_AUDIT.json', report)
    print(json.dumps(dict(status='PASS', contracts=contract, comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
