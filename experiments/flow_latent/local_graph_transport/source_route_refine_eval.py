"""Evaluate frozen source/route refinements; labels never change their scores.

Fit-mixture thresholds are the detector operating points. Two matched-budget
policies use evaluation labels and are explicitly oracle diagnostics.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import pairwise_within, source_bootstrap
from .run_capture import CACHE, file_hash
from .source_route_refine import METHODS, PRIMARY


BASELINE = 'old_native'
PACK_FIELDS = ('token_id', 'target', 'source_index', 'answer_index', 'unit_index')
CONTROLS = ('old_native', 'mean_cdf_robust', 'union_native', 'union_chain', 'union_rewired')


def frozen_inputs(directory, phase):
    """Check every declared hash and required binding before any label access."""
    frozen_path = directory / f'{phase}_FREEZE.json'
    frozen = json.loads(frozen_path.read_text())
    required = (f'{phase}_scores.npz', f'{phase}_pack.npz',
                f'{phase}_metadata.json', 'REFINE_REFERENCE.json')
    hashes = frozen['hashes']
    if not set(required).issubset(hashes):
        raise ValueError('Freeze must bind scores, pack, metadata and reference')
    for name, expected in hashes.items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f'Freeze path escapes output directory: {name}')
        if file_hash(path) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    with np.load(directory / required[0], allow_pickle=False) as saved:
        scores = {name: saved[name].copy() for name in METHODS}
    with np.load(directory / required[1], allow_pickle=False) as saved:
        if set(saved.files) != set(PACK_FIELDS):
            raise ValueError('Evaluation pack must contain only token identity/group fields')
        pack = {name: saved[name].copy() for name in PACK_FIELDS}
    metadata = json.loads((directory / required[2]).read_text())
    reference = json.loads((directory / required[3]).read_text())
    thresholds = reference['thresholds']
    thresholds = thresholds.get('fit95', thresholds)
    thresholds = {name: float(thresholds[name]) for name in METHODS}
    verify_alignment(pack, scores, metadata, thresholds)
    binding = dict(freeze_sha256=file_hash(frozen_path), artifact_sha256=hashes)
    return pack, scores, metadata, thresholds, binding


def verify_alignment(pack, scores, metadata, thresholds):
    """Require complete, contiguous answer blocks and finite independent scores."""
    count = len(pack['token_id'])
    if count == 0 or any(value.shape != (count,) for value in pack.values()):
        raise ValueError('Pack fields must cover a nonempty common token axis')
    if any(value.shape != (count,) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError('Every frozen method must cover all tokens with finite scores')
    if not all(np.isfinite(value) for value in thresholds.values()):
        raise ValueError('Every method requires a finite fit-mixture threshold')
    cursor, identities, source_groups = 0, set(), {}
    for index, record in enumerate(metadata['records']):
        start, stop = record['packed_start'], record['packed_stop']
        if start != cursor or stop <= start or stop > count or record['id'] in identities:
            raise ValueError('Metadata must partition tokens into unique contiguous answers')
        if not np.all(pack['answer_index'][start:stop] == index):
            raise ValueError(f"{record['id']}: answer_index must match reordered records")
        targets = pack['target'][start:stop]
        if (targets < 0).any() or not (np.diff(targets) > 0).all():
            raise ValueError(f"{record['id']}: original targets must be strictly increasing")
        if len(np.unique(pack['source_index'][start:stop])) != 1:
            raise ValueError(f"{record['id']}: answer contains multiple packed sources")
        source = record['source_id']
        group = int(pack['source_index'][start])
        if source in source_groups and source_groups[source] != group:
            raise ValueError(f'{source}: one source must have one bootstrap group')
        if source not in source_groups and group in source_groups.values():
            raise ValueError('Different sources cannot share a bootstrap group')
        source_groups[source] = group
        identities.add(record['id'])
        cursor = stop
    if cursor != count:
        raise ValueError('Metadata leaves token rows uncovered')


def answer_masks(pack, records):
    """Healthy-prefix first-error ranking excludes all positions after first error."""
    clean = np.zeros(len(pack['labels']), dtype=bool)
    first = np.zeros_like(clean)
    answers = []
    for record in records:
        start, stop = record['packed_start'], record['packed_stop']
        errors = np.flatnonzero(pack['labels'][start:stop])
        onset = start + int(errors[0]) if len(errors) else None
        clean[start:onset if onset is not None else stop] = True
        if onset is not None:
            first[onset] = True
        answers.append(dict(id=record['id'], start=start, stop=stop, first=onset))
    if not np.array_equal(first, pack['firsts']):
        raise ValueError('Official first errors disagree with packed token order')
    return clean, first, answers


def alarm_counts(pack, scores, threshold, clean, first, answers):
    """Report token and answer alarms, including healthy-prefix first-error recall."""
    alarm = scores > threshold
    positive = pack['labels'].astype(bool)
    tp, fp = int((alarm & positive).sum()), int((alarm & ~positive).sum())
    normal_answers = [row for row in answers if row['first'] is None]
    error_answers = [row for row in answers if row['first'] is not None]
    normal_alarm = sum(bool(alarm[row['start']:row['stop']].any()) for row in normal_answers)
    prior_alarm = sum(bool(alarm[row['start']:row['first']].any()) for row in error_answers)
    timely = sum(bool(alarm[row['first']] and not alarm[row['start']:row['first']].any())
                 for row in error_answers)
    return dict(threshold=float(threshold), rule='score > threshold', tp=tp, fp=fp,
        fn=int(positive.sum()) - tp, tn=int((~positive).sum()) - fp,
        token_recall=tp / int(positive.sum()) if positive.any() else None,
        token_fpr=fp / int((~positive).sum()) if (~positive).any() else None,
        normal_answers=len(normal_answers), normal_answer_alarms=int(normal_alarm),
        normal_answer_alarm_rate=normal_alarm / len(normal_answers) if normal_answers else None,
        error_answers=len(error_answers), first_errors_detected=int(alarm[first].sum()),
        first_errors_without_prior_alarm=int(timely),
        error_answers_with_prior_false_alarm=int(prior_alarm),
        clean_normal_tokens=int(clean.sum()), clean_normal_false_alarms=int(alarm[clean].sum()),
        onsets=int(pack['onsets'].sum()), onsets_detected=int(alarm[pack['onsets']].sum()))


def threshold_for_budget(normal_values, budget):
    """Strict threshold allowing at most budget; ties can leave budget unused."""
    values = np.sort(np.asarray(normal_values, dtype=np.float64))[::-1]
    if not len(values):
        return None
    if not 0 <= budget <= len(values):
        raise ValueError('False-alarm budget must be between zero and population size')
    if budget == len(values):
        return float(np.nextafter(values[-1], -np.inf))
    return float(values[budget])


def method_report(pack, values, fit_threshold, clean, first, answers, budgets):
    """All rankings and fixed/oracle policies use the same frozen token scores."""
    healthy_first = clean | first
    onset_mask = (pack['labels'] == 0) | pack['onsets']
    result = dict(all_tokens=ranking(pack['labels'], values),
        within_answer_auroc=pairwise_within(pack['labels'], values, pack['answer_index']),
        strict_first_error=ranking(first[healthy_first].astype(int), values[healthy_first]),
        onsets_vs_normal=ranking(pack['onsets'][onset_mask].astype(int), values[onset_mask]))
    fixed = alarm_counts(pack, values, fit_threshold, clean, first, answers)
    result['fit_mixture95'] = dict(fixed, label_assisted=False, calibrated_normal_fpr=False)
    normal_maxima = [values[row['start']:row['stop']].max()
                     for row in answers if row['first'] is None]
    populations = dict(normal_token_budget=values[pack['labels'] == 0],
                       normal_answer_budget=normal_maxima)
    for policy, population in populations.items():
        cutoff = threshold_for_budget(population, budgets[policy])
        if cutoff is None:
            result[policy] = dict(status='unavailable_no_normal_population')
            continue
        counts = alarm_counts(pack, values, cutoff, clean, first, answers)
        result[policy] = dict(counts, label_assisted=True, oracle_diagnostic=True,
            independent_calibration=False, allowed_budget=budgets[policy],
            budget_baseline=BASELINE, ties='strict threshold may use less than allowed budget')
    return result


def transitions(labels, baseline, candidate, baseline_threshold, candidate_threshold):
    """Count changed alarms; rankings and token labels remain untouched."""
    old, new = baseline > baseline_threshold, candidate > candidate_threshold
    positive = labels.astype(bool)
    added, removed = ~old & new, old & ~new
    return dict(new_tp=int((added & positive).sum()), new_fp=int((added & ~positive).sum()),
        lost_tp=int((removed & positive).sum()), removed_fp=int((removed & ~positive).sum()),
        baseline_threshold=float(baseline_threshold), candidate_threshold=float(candidate_threshold))


def write_tokens(path, pack, records, scores, thresholds):
    """Export all scored tokens; text is read from the original response only."""
    columns = ['id', 'target', 'token_id', 'word', 'label', 'onset', 'first_error']
    columns += [column for name in METHODS for column in (name, name + '__alarm')]
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            response = json.loads((CACHE / record['directory'] / 'response.json').read_text())
            text = response.get('token_text')
            for index in range(record['packed_start'], record['packed_stop']):
                target = int(pack['target'][index])
                token_id = int(pack['token_id'][index])
                if int(response['answer_ids'][target]) != token_id:
                    raise ValueError(f"{record['id']}: CSV original token identity differs")
                word = text[target] if text is not None else str(token_id)
                row = dict(id=record['id'], target=target, token_id=token_id, word=word,
                    label=int(pack['labels'][index]), onset=int(pack['onsets'][index]),
                    first_error=int(pack['firsts'][index]))
                for name in METHODS:
                    value = float(scores[name][index])
                    row[name], row[name + '__alarm'] = value, int(value > thresholds[name])
                writer.writerow(row)


def build_report(pack, scores, records, thresholds, binding, phase):
    """Use a fixed primary and all declared controls, without method selection."""
    clean, first, answers = answer_masks(pack, records)
    baseline = alarm_counts(pack, scores[BASELINE], thresholds[BASELINE], clean, first, answers)
    budgets = dict(normal_token_budget=baseline['fp'], normal_answer_budget=baseline['normal_answer_alarms'])
    methods = {name: method_report(pack, scores[name], thresholds[name], clean, first, answers, budgets)
               for name in METHODS}
    comparisons = {name: source_bootstrap(pack, scores[PRIMARY], scores[name], repeats=300)
                   for name in CONTROLS}
    changed = {}
    for policy in ('fit_mixture95', 'normal_token_budget', 'normal_answer_budget'):
        old, new = methods[BASELINE][policy], methods[PRIMARY][policy]
        if 'threshold' in old and 'threshold' in new:
            changed[policy] = transitions(pack['labels'], scores[BASELINE], scores[PRIMARY],
                                           old['threshold'], new['threshold'])
    return dict(phase=phase, primary=PRIMARY, baseline=BASELINE, selection=False,
        natural_labels_used_for_score_fit=False, historical_exposure=True, offline=True,
        strict_first_error_scope='normal answers plus healthy prefixes and the first error of error answers',
        methods=methods, primary_source_bootstrap=comparisons, baseline_to_primary_transitions=changed,
        counts=dict(answers=len(records), sources=len(np.unique(pack['source_index'])),
                    tokens=len(pack['labels']), positives=int(pack['labels'].sum()),
                    first_errors=int(first.sum()), onsets=int(pack['onsets'].sum())),
        binding=binding, oracle_policies='label-assisted diagnostic; no independently calibrated deployment claim')


def evaluate(directory, phase):
    """Read labels only after the complete phase freeze has passed hash checks."""
    directory = Path(directory)
    report_path = directory / f'{phase}_evaluation.json'
    token_path = directory / f'{phase}_tokens.csv'
    if report_path.exists() or token_path.exists():
        raise FileExistsError('Evaluation artifacts exist; use a fresh output directory')
    pack, scores, metadata, thresholds, binding = frozen_inputs(directory, phase)
    pack.update(evaluation_labels(CACHE, pack, metadata))
    report = build_report(pack, scores, metadata['records'], thresholds, binding, phase)
    write_tokens(token_path, pack, metadata['records'], scores, thresholds)
    report['token_csv_sha256'] = file_hash(token_path)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    arguments = parser.parse_args()
    result = evaluate(arguments.directory, arguments.phase)
    print(json.dumps(dict(phase=result['phase'], counts=result['counts'],
                         primary=result['methods'][PRIMARY]['all_tokens']), ensure_ascii=False))
