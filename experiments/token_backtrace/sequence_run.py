"""Fit, freeze and evaluate source-conditioned sequence models on 8+6 controls."""

import argparse
import csv
import html
import shutil
from pathlib import Path

import numpy as np

from .sequence import FEATURES, INPUTS, STATES, SequenceModel, fit_sequence
from .sequence_data import (PACKS, answer_weights, controls, features, fit_reference, load_pack,
                            read_json, select_reference, selected_controls, transform_answer,
                            weighted_quantile, write_json)
from .span_metrics import character_counts
from experiments.context_response.restore import cached_baselines, original_threshold
from experiments.native_support.evaluate import annotation_targets, ranking
from experiments.native_support.ragtruth_benchmark.data import TASKS, annotations
from experiments.probabilistic_detection.evaluation import source_bootstrap
from experiments.unsupervised_graph.scalar import fit_ranks, rank_scores, scalar_scores


VARIANTS = ('source_sequence', 'support_hmm', 'homogeneous_four', 'unconditioned_four')
METHODS = (*VARIANTS, 'iid', 'fixed_reference', 'historical_fixed')


def settings(args):
    return dict(features=list(FEATURES), inputs=list(INPUTS), states=list(STATES),
                seeds=args.seeds, iterations=args.iterations, transition_steps=args.transition_steps,
                shrinkage=args.shrinkage, margin=args.margin, penalty=args.penalty,
                fit_sources=args.fit_sources, dev_sources=args.dev_sources, selection_seed=42,
                threshold_quantile=args.quantile, source_balanced=True, offline=True,
                anchored_emissions=args.anchored_emissions, unit_support=args.unit_support,
                packs=str(args.packs.resolve()), controls=[row['id'] for row in controls()],
                checkpoint_selection='unlabelled_dev_source_balanced_loglik', labels_used=False,
                covariance_floor=.05, initial_stay=.9, nuisance_ridge=1e-6,
                scope='exposed controls: source-isolated development diagnostic, not blind test')


def reference_scores(answers, references):
    return [rank_scores(scalar_scores(row['pack']), references)['fixed_unsupervised'] for row in answers]


def threshold(scores, answers, quantile):
    weights = np.concatenate([np.full(len(values), weight)
                              for values, weight in zip(scores, answer_weights(answers))])
    return weighted_quantile(np.concatenate(scores), weights, quantile)


def fit_variant(args, task, name, train, development, reference):
    train_values = [transform_answer(row, reference) for row in train]
    dev_values = [transform_answer(row, reference) for row in development]
    seeds = args.seeds if name == 'source_sequence' else args.seeds[:1]
    selected, best = None, -np.inf
    for seed in seeds:
        print(f'{task} {name} seed={seed}: fitting {len(train)} answers', flush=True)
        model, history = fit_sequence(train_values, dev_values, answer_weights(train),
            answer_weights(development), states=2 if name == 'support_hmm' else 4,
            driven=name not in ('support_hmm', 'homogeneous_four'), seed=seed,
            iterations=args.iterations, transition_steps=args.transition_steps,
            shrinkage=args.shrinkage, margin=args.margin, penalty=args.penalty,
            anchored=args.anchored_emissions)
        best_iteration = max(history, key=lambda row: row['dev_loglik'])
        folder = args.output / task / name
        write_json(folder / f'seed_{seed}.json', dict(seed=seed, history=history, selected=best_iteration))
        np.savez(folder / f'seed_{seed}.npz', **model.arrays())
        if best_iteration['dev_loglik'] > best:
            best = best_iteration['dev_loglik']
            selected = dict(seed=seed, iteration=best_iteration['iteration'], dev_loglik=best)
    write_json(args.output / task / name / 'selection.json', selected)
    with np.load(args.output / task / name / f"seed_{selected['seed']}.npz") as saved:
        return SequenceModel(**{key: saved[key] for key in saved.files})


def fit_task(args, task, rows):
    pack, metadata = load_pack(task, 'train', args.packs)
    excluded = {row['source_id'] for row in rows}
    train, development = select_reference(pack, metadata, excluded, args.fit_sources, args.dev_sources)
    if args.unit_support:
        for row in train + development:
            row['values'] = features(row['pack'], unit_support=True)
    folder = args.output / task
    write_json(folder / 'reference_sources.json', dict(excluded=sorted(excluded),
               fit=[row['record'] for row in train], dev=[row['record'] for row in development]))
    reference = fit_reference(train)
    unconditioned = fit_reference(train, conditioned=False)
    write_json(folder / 'reference.json', reference)
    write_json(folder / 'unconditioned_reference.json', unconditioned)
    thresholds = {}
    for name in VARIANTS:
        transform = unconditioned if name == 'unconditioned_four' else reference
        model = fit_variant(args, task, name, train, development, transform)
        dev_scores = [model.score(transform_answer(row, transform)) for row in development]
        thresholds[name] = threshold([row['risk'] for row in dev_scores], development, args.quantile)
        if name == 'source_sequence':
            thresholds['iid'] = threshold([row['iid'] for row in dev_scores], development, args.quantile)
    fit_pack = {key: np.concatenate([row['pack'][key] for row in train]) for key in ('context', 'observations')}
    ranks = fit_ranks(scalar_scores(fit_pack))
    np.savez(folder / 'fixed_reference.npz', **ranks)
    thresholds['fixed_reference'] = threshold(reference_scores(development, ranks), development, args.quantile)
    thresholds['historical_fixed'] = original_threshold(task)
    write_json(folder / 'thresholds.json', thresholds)


def fit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', settings(args))
    source = args.output / 'executed_source'
    source.mkdir()
    for name in ('sequence.py', 'sequence_data.py', 'sequence_run.py'):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    rows = controls()
    for task in TASKS:
        fit_task(args, task, rows)
    write_json(args.output / 'fit_complete.json', dict(tasks=list(TASKS), labels_used=False))


def score_task(args, task, rows, historical):
    folder = args.output / task
    unit_support = read_json(args.output / 'protocol.json').get('unit_support', False)
    reference = read_json(folder / 'reference.json')
    unconditioned = read_json(folder / 'unconditioned_reference.json')
    models = {}
    for name in VARIANTS:
        selection = read_json(folder / name / 'selection.json')
        with np.load(folder / name / f"seed_{selection['seed']}.npz") as saved:
            models[name] = SequenceModel(**{key: saved[key] for key in saved.files})
    with np.load(folder / 'fixed_reference.npz') as saved:
        ranks = {key: saved[key] for key in saved.files}
    records = []
    for row in selected_controls(task, rows, args.packs):
        if unit_support:
            row['values'] = features(row['pack'], unit_support=True)
        scores = dict(target=row['pack']['target'], token_id=row['pack']['token_id'])
        for name, model in models.items():
            transform = unconditioned if name == 'unconditioned_four' else reference
            prediction = model.score(transform_answer(row, transform))
            scores[name] = prediction['risk']
            if name == 'source_sequence':
                scores.update({key: prediction[key] for key in ('iid', 'posterior', 'enter', 'exit', 'path')})
        scores['fixed_reference'] = reference_scores([row], ranks)[0]
        scores['historical_fixed'] = historical[row['record']['id']][scores['target']]
        assert all(np.isfinite(value).all() for value in scores.values())
        identity = row['record']['id']
        np.savez(folder / f'{identity}_predictions.npz', **scores)
        records.append(dict(row['record'], root=row['root'], score_file=f'{task}/{identity}_predictions.npz'))
    return records


def score(args):
    read_json(args.output / 'fit_complete.json')
    protocol = read_json(args.output / 'protocol.json')
    args.packs = Path(protocol['packs'])
    rows = controls()
    historical = cached_baselines({row['id'] for row in rows})
    records = [record for task in TASKS for record in score_task(args, task, rows, historical)]
    assert {row['id'] for row in records} == {row['id'] for row in rows}
    quantile = protocol['threshold_quantile']
    write_json(args.output / 'predictions_manifest.json', dict(records=records, methods=list(METHODS),
        labels_used=False, threshold_rule=f'unlabelled_dev_mixture_source_balanced_{quantile:g}q_strict_gt',
        historical_rule='original_full_reference_and_original_mixed_threshold', frozen=True))
    print(f'Frozen predictions for {len(records)} answers before evaluation', flush=True)


def case_metrics(labels, onsets, firsts, scores, alarm, chars):
    result = ranking(labels, scores)
    tp, fp, fn = int((alarm & (labels == 1)).sum()), int((alarm & (labels == 0)).sum()), int((~alarm & (labels == 1)).sum())
    continuation = labels.astype(bool) & ~onsets
    recovery = np.r_[False, labels[:-1] == 1] & (labels == 0)
    result.update(tp=tp, fp=fp, fn=fn, onset_hits=int(alarm[onsets].sum()), onsets=int(onsets.sum()),
                  first_hits=int(alarm[firsts].sum()), firsts=int(firsts.sum()),
                  continuation_hits=int(alarm[continuation].sum()), continuations=int(continuation.sum()),
                  recovery_false_alarms=int(alarm[recovery].sum()), recoveries=int(recovery.sum()),
                  late_tp=int((alarm & (labels == 1) & (np.arange(len(labels)) >= len(labels) / 2)).sum()),
                  normal_answer_alarm=bool(alarm.any()) if not labels.any() else None,
                  character_counts=chars.tolist())
    return result


def evaluate_case(args, record, truth):
    response = read_json(Path(record['root']) / record['directory'] / 'response.json')
    with np.load(args.output / record['score_file']) as saved:
        values = {key: saved[key] for key in saved.files}
    labels, onsets, firsts, valid = annotation_targets(truth, record['tokens'], record['id'])
    target = values['target']
    assert np.array_equal(target, np.flatnonzero(valid))
    assert np.array_equal(values['token_id'], np.asarray(truth['token_ids'])[target])
    thresholds = read_json(args.output / record['task'] / 'thresholds.json')
    metrics, token_rows = {}, []
    for name in METHODS:
        alarm = values[name] > thresholds[name]
        offsets = np.asarray(response['offsets'])[target]
        chars = character_counts(response['text'], truth['character_spans'], offsets, alarm)
        metrics[name] = case_metrics(labels[target], onsets[target], firsts[target], values[name], alarm, chars)
        metrics[name]['threshold'] = thresholds[name]
    for index, original in enumerate(target):
        entry = dict(id=record['id'], task=record['task'], cohort=record['cohort'], target=int(original),
                     text=response['token_text'][original], gold=int(labels[original]),
                     onset=bool(onsets[original]), first=bool(firsts[original]),
                     state=int(values['path'][index]), enter=float(values['enter'][index]), exit=float(values['exit'][index]))
        for name in METHODS:
            entry[name] = float(values[name][index])
            entry[name + '_alarm'] = bool(values[name][index] > thresholds[name])
        entry.update({state: float(values['posterior'][index, number]) for number, state in enumerate(STATES)})
        token_rows.append(entry)
    return dict(record=record, metrics=metrics), token_rows


def aggregate(cases, tokens):
    result = {}
    groups = ('all', 'original8', 'extension6', *TASKS)
    for group in groups:
        selected = [case for case in cases if group in ('all', case['record']['cohort'], case['record']['task'])]
        rows = [row for row in tokens if group in ('all', row['cohort'], row['task'])]
        methods = {}
        for name in METHODS:
            summary = ranking(np.array([row['gold'] for row in rows]), np.array([row[name] for row in rows]))
            for key in ('tp', 'fp', 'fn', 'onset_hits', 'onsets', 'first_hits', 'firsts',
                        'continuation_hits', 'continuations', 'recovery_false_alarms', 'recoveries', 'late_tp'):
                summary[key] = sum(case['metrics'][name][key] for case in selected)
            counts = np.sum([case['metrics'][name]['character_counts'] for case in selected], axis=0)
            tp, fp, fn, spans, any_hit, full_hit = map(int, counts)
            summary.update(character_iou=tp / max(1, tp + fp + fn), spans=spans,
                           spans_any=any_hit, spans_full=full_hit,
                           normal_answers=sum(case['metrics'][name]['normal_answer_alarm'] is not None for case in selected),
                           normal_answer_alarms=sum(case['metrics'][name]['normal_answer_alarm'] is True for case in selected))
            methods[name] = summary
        result[group] = methods
    return result


def write_view(output, cases, tokens):
    sections = []
    for case in cases:
        identity = case['record']['id']
        pieces = []
        for row in (row for row in tokens if row['id'] == identity):
            alarm = row['source_sequence_alarm']
            category = ('tp' if row['gold'] else 'fp') if alarm else ('fn' if row['gold'] else 'tn')
            tooltip = f"token={row['target']} gold={row['gold']} state={STATES[row['state']]} "
            tooltip += ' '.join(f'{name}={row[name]:.5f}/{int(row[name + "_alarm"])}' for name in METHODS)
            pieces.append(f'<span class="{category}" title="{html.escape(tooltip, quote=True)}">{html.escape(row["text"])}</span>')
        sections.append(f'<h2>{identity} {case["record"]["task"]} {case["record"]["cohort"]}</h2><pre>{"".join(pieces)}</pre>')
    style = 'body{max-width:1100px;margin:30px auto;font-family:sans-serif}pre{white-space:pre-wrap;line-height:2}.tp{background:#a8e6b0}.fp{background:#f4db9c}.fn{background:#f5aaaa}span:hover{outline:1px solid black}'
    output.write_text('<!doctype html><meta charset="utf-8"><title>来源序列完整token诊断</title><style>' + style +
                      '</style><h1>四状态来源序列：全部token</h1><p>绿=TP，黄=FP，红=FN，白=TN；悬停显示全部方法分数/报警与状态。阈值均在评价前冻结。这批已暴露样本是开发诊断。</p>' + ''.join(sections))


def evaluate(args):
    manifest = read_json(args.output / 'predictions_manifest.json')
    cases, tokens = [], []
    root = Path(manifest['records'][0]['root'])
    truth = annotations(root, read_json(root / 'manifest.json'), manifest['records'])
    for record in manifest['records']:
        case, rows = evaluate_case(args, record, truth[record['id']])
        cases.append(case)
        tokens.extend(rows)
    write_json(args.output / 'case_results.json', cases)
    write_json(args.output / 'results.json', aggregate(cases, tokens))
    state_summary = {}
    for gold in (0, 1):
        selected = [row for row in tokens if row['gold'] == gold]
        state_summary[str(gold)] = dict(tokens=len(selected),
            mean_posterior={state: float(np.mean([row[state] for row in selected])) for state in STATES},
            viterbi_counts={state: sum(row['state'] == index for row in selected)
                            for index, state in enumerate(STATES)})
    write_json(args.output / 'state_diagnostics.json', state_summary)
    source_index = {case['record']['id']: index for index, case in enumerate(cases)}
    boot_pack = dict(labels=np.array([row['gold'] for row in tokens]),
                     source_index=np.array([source_index[row['id']] for row in tokens]))
    candidate = np.array([row['source_sequence'] for row in tokens])
    baseline = np.array([row['fixed_reference'] for row in tokens])
    write_json(args.output / 'source_bootstrap.json', source_bootstrap(boot_pack, candidate, baseline))
    with (args.output / 'tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tokens[0]))
        writer.writeheader()
        writer.writerows(tokens)
    write_view(args.output / 'tokens.html', cases, tokens)
    print(f'Evaluated all {len(tokens)} valid tokens against official gold', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit', 'score', 'evaluate', 'run-controls'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packs', type=Path, default=PACKS,
                        help='fit-time pack root; score consumes the saved protocol path')
    parser.add_argument('--fit-sources', type=int, default=32)
    parser.add_argument('--dev-sources', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 17, 314])
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--transition-steps', type=int, default=30)
    parser.add_argument('--shrinkage', type=float, default=.2)
    parser.add_argument('--margin', type=float, default=.2)
    parser.add_argument('--penalty', type=float, default=.01)
    parser.add_argument('--quantile', type=float, default=.95)
    parser.add_argument('--anchored-emissions', action='store_true',
                        help='structured mean subspace preserving source likelihood-ratio direction')
    parser.add_argument('--unit-support', action='store_true',
                        help='use existing full/local unit means for the two source coordinates')
    args = parser.parse_args()
    for stage in ('fit', 'score', 'evaluate') if args.stage == 'run-controls' else (args.stage,):
        globals()[stage](args)


if __name__ == '__main__':
    main()
