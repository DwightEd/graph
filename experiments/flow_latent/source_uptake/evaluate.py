"""Evaluate immutable all-token scores; annotations are opened after verification."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re

import numpy as np

from state_audit.dataset import Example
from state_audit.dataset.jsonl import validate
from experiments.native_support.ragtruth import aligned_labels


VARIANTS = ('single', 'mean', 'ordered', 'shuffled', 'drop_source', 'linear')
SEEDS = (42, 123, 2026)
OBSERVED_SUFFIXES = ('gap', 'logp_gap')
PRIOR_KEYS = ('prior_source_pair_unit_mean', 'prior_source_local_unit_mean', 'prior_raw_route')
BASELINES = ('entropy', 'surprisal') + PRIOR_KEYS
ID_FIELD = re.compile(r'"id"\s*:\s*("[^"\\]*"|\d+)')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def score_keys(suffixes):
    return [f'{variant}_seed{seed}_{suffix}' for variant in VARIANTS
            for seed in SEEDS for suffix in suffixes]


def resolve_file(root, name):
    path = Path(name)
    return (path if path.is_absolute() else root / path).resolve()


def verify_hashes(root, hashes, verified):
    for name, expected in hashes.items():
        path = resolve_file(root, name)
        actual = verified.get(path)
        if actual is None:
            actual = sha256(path)
            verified[path] = actual
        if actual != expected:
            raise ValueError(f'Frozen SHA256 mismatch: {path}')


def verify_freeze(root, roster_path):
    """Check complete answer coverage and immutable input/model binding first."""
    freeze_path = root / 'FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    if freeze['status'] != 'COMPLETE':
        raise ValueError('Evaluation requires a COMPLETE score freeze')
    bound_roster = resolve_file(root, freeze['roster']['path'])
    if bound_roster != roster_path.resolve():
        raise ValueError('The supplied roster differs from the frozen roster path')

    verified = {}
    verify_hashes(root, {str(bound_roster): freeze['roster']['sha256']}, verified)
    roster = json.loads(bound_roster.read_text())
    annotation_path = Path(roster['input_paths']['annotation_file_for_later_frozen_evaluation']).resolve()
    early_hashes = list(freeze['files']) + list(freeze['checkpoints']) + list(roster['input_sha256'])
    if any(resolve_file(root, path) == annotation_path for path in early_hashes):
        raise ValueError('Annotations cannot enter pre-evaluation input hashing')
    verify_hashes(root, freeze['files'], verified)
    verify_hashes(root, freeze['checkpoints'], verified)
    expected_models = {f'{variant}_seed{seed}' for variant in VARIANTS for seed in SEEDS}
    frozen_models = {Path(path).parent.name for path in freeze['checkpoints']}
    if not expected_models.issubset(frozen_models):
        raise ValueError('Freeze does not bind all six views and three checkpoint seeds')

    verify_hashes(root, roster['input_sha256'], verified)
    frozen_files = {resolve_file(root, path) for path in freeze['files']}
    for record in roster['natural']['records']:
        directory = root / 'answers' / record['id']
        required = {(directory / name).resolve() for name in ('data.json', 'scores.npz')}
        if not required.issubset(frozen_files):
            raise ValueError(f'{record["id"]}: answer data/scores are absent from freeze')
    return freeze, roster, verified


def metric_code_paths():
    """Pin the executing evaluator and exact official alignment implementation."""
    token_spans = aligned_labels.__globals__['token_spans']
    return {Path(__file__).resolve(), Path(aligned_labels.__code__.co_filename).resolve(),
            Path(validate.__code__.co_filename).resolve(), Path(inspect.getfile(Example)).resolve(),
            Path(token_spans.__code__.co_filename).resolve()}


def verify_evaluation_protocol(root, freeze, verified, draws, seed):
    path = (root / 'EVALUATION_PROTOCOL.json').resolve()
    frozen_files = {resolve_file(root, name) for name in freeze['files']}
    if path not in frozen_files:
        raise ValueError('Evaluation protocol is absent from the score freeze')
    protocol = json.loads(path.read_text())
    expected = dict(primary_key='ordered_seed42_gap', fixed_replication_seeds=[123, 2026],
                    threshold=0, bootstrap_draws=draws, bootstrap_seed=seed)
    for field, value in expected.items():
        if protocol[field] != value:
            raise ValueError(f'Evaluation protocol mismatch: {field}')
    named_code = {resolve_file(root, name) for name in protocol['metric_code_hashes']}
    if not metric_code_paths().issubset(named_code) or not named_code.issubset(frozen_files):
        raise ValueError('Evaluation/alignment code is not fully pinned in protocol and freeze')
    verify_hashes(root, protocol['metric_code_hashes'], verified)
    return protocol


def validate_answer(record, data, scores, response):
    """Token/offset identity and every planned score are scientific contracts."""
    if data['response_id'] != record['id'] or data['source_id'] != record['source_id']:
        raise ValueError(f'{record["id"]}: response/source identity differs')
    if data['text'] != response['text'] or data['offsets'] != response['offsets']:
        raise ValueError(f'{record["id"]}: frozen text or offsets differ from prepared response')
    if data['special_ids'] != response['special_ids']:
        raise ValueError(f'{record["id"]}: special-token IDs differ')
    count = record['tokens']
    if len(response['answer_ids']) != count:
        raise ValueError(f'{record["id"]}: roster token count differs')
    np.testing.assert_array_equal(scores['actual_ids'], response['answer_ids'])
    if scores['candidate_ids'].shape != (count, 32):
        raise ValueError(f'{record["id"]}: native top32 coverage differs')
    if not np.all(np.diff(np.sort(scores['candidate_ids'], axis=1), axis=1) > 0):
        raise ValueError(f'{record["id"]}: duplicate native candidates')
    np.testing.assert_array_equal(scores['greedy_ids'], scores['candidate_ids'][:, 0])
    membership = (scores['candidate_ids'] == scores['actual_ids'][:, None]).any(axis=1)
    np.testing.assert_array_equal(scores['actual_in_topk'], membership)
    validate_native_and_rivals(record['id'], scores, count)
    for key in score_keys(OBSERVED_SUFFIXES + ('native_gap',)) + ['entropy', 'surprisal']:
        if scores[key].shape != (count,) or not np.isfinite(scores[key]).all():
            raise ValueError(f'{record["id"]}: incomplete/nonfinite {key}')
    for key in PRIOR_KEYS:
        if key in scores and (scores[key].shape != (count,) or not np.isfinite(scores[key]).all()):
            raise ValueError(f'{record["id"]}: incomplete/nonfinite prior scalar {key}')


def validate_native_and_rivals(identity, scores, count):
    logp = scores['candidate_logp']
    if logp.shape != (count, 32) or not np.isfinite(logp).all():
        raise ValueError(f'{identity}: incomplete/nonfinite native candidate logp')
    if scores['actual_logp'].shape != (count,) or not np.isfinite(scores['actual_logp']).all():
        raise ValueError(f'{identity}: incomplete/nonfinite actual logp')
    if np.any(np.diff(logp, axis=1) > 0):
        raise ValueError(f'{identity}: native top32 logp is not descending')
    if np.any(scores['entropy'] < 0):
        raise ValueError(f'{identity}: native entropy is negative')
    np.testing.assert_allclose(scores['surprisal'], -scores['actual_logp'], rtol=0, atol=1e-6)
    matches = scores['candidate_ids'] == scores['actual_ids'][:, None]
    rows, columns = np.nonzero(matches)
    np.testing.assert_allclose(scores['actual_logp'][rows], logp[rows, columns], rtol=0, atol=1e-6)
    for key in score_keys(OBSERVED_SUFFIXES + ('native_gap',)):
        rival = scores[key.removesuffix('_gap') + '_rival_ids']
        if rival.shape != (count,):
            raise ValueError(f'{identity}: incomplete {key} rival IDs')
        in_top32 = (scores['candidate_ids'] == rival[:, None]).any(axis=1)
        selected = scores['greedy_ids'] if key.endswith('_native_gap') else scores['actual_ids']
        if not in_top32.all() or np.any(rival == selected):
            raise ValueError(f'{identity}: {key} rival must be a distinct native top32 candidate')


def load_answers(root, roster, verified):
    answers = []
    annotation_path = Path(roster['input_paths']['annotation_file_for_later_frozen_evaluation']).resolve()
    for record in roster['natural']['records']:
        directory = root / 'answers' / record['id']
        data = json.loads((directory / 'data.json').read_text())
        with np.load(directory / 'scores.npz', allow_pickle=False) as saved:
            scores = {key: saved[key] for key in saved.files}
        response_path = Path(record['response_cache_path'])
        response = json.loads(response_path.read_text())
        if any(resolve_file(root, path) == annotation_path for path in data['input_hashes']):
            raise ValueError('Annotations cannot enter answer input hashing')
        verify_hashes(root, data['input_hashes'], verified)
        required_inputs = {str(Path(record[key]).resolve()) for key in
                           ('response_cache_path', 'source_cache_path')}
        named_inputs = {str(resolve_file(root, path)) for path in data['input_hashes']}
        if not required_inputs.issubset(named_inputs):
            raise ValueError(f'{record["id"]}: prepared source/response input hashes are missing')
        validate_answer(record, data, scores, response)
        answers.append(dict(record=record, data=data, scores=scores))
    if len(answers) != roster['natural']['answers']:
        raise ValueError('Frozen answer count differs from roster')
    if sum(len(answer['scores']['actual_ids']) for answer in answers) != roster['natural']['tokens']:
        raise ValueError('Frozen all-token count differs from roster')
    return answers


def load_annotations(roster, answers):
    """Only allow-listed official rows are parsed, after freeze verification."""
    wanted = {answer['record']['id']: answer for answer in answers}
    annotations = {}
    path = Path(roster['input_paths']['annotation_file_for_later_frozen_evaluation'])
    with path.open() as stream:
        for line in stream:
            match = ID_FIELD.search(line)
            identity = str(json.loads(match.group(1))) if match else None
            if identity not in wanted:
                continue
            row = json.loads(line)
            answer = wanted[identity]
            data = answer['data']
            if row['labels'] is None:
                raise ValueError(f'{identity}: missing labels are not reviewed negatives')
            if row['response'] != data['text'] or str(row['source_id']) != data['source_id']:
                raise ValueError(f'{identity}: official answer changed after score freeze')
            example = Example(identity, data['source_id'], '', response=data['text'], labels=row['labels'])
            validate(example)
            encoded = dict(input_ids=answer['scores']['actual_ids'], offset_mapping=data['offsets'])
            annotations[identity] = aligned_labels(example, encoded, data['special_ids'])
    if set(annotations) != set(wanted):
        raise ValueError('Official annotations do not cover every frozen answer')
    for answer in answers:
        annotation = annotations[answer['record']['id']]
        answer['labels'] = np.asarray(annotation['labels'], dtype=bool)
        answer['onsets'] = np.asarray(annotation['span_onsets'], dtype=bool)
        answer['valid'] = np.asarray(annotation['valid_tokens'], dtype=bool)


def ranking(labels, scores):
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind='stable')
    ordered_scores = np.asarray(scores)[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(order) - 1] if len(order) else np.array([], int)
    return order, ends, np.asarray(labels, dtype=bool)[order]


def rank_metrics(ranked, weights=None):
    """Exact tied-score AUC and non-interpolated AP; optional source multiplicity."""
    order, ends, positive = ranked
    weight = np.ones(len(order)) if weights is None else np.asarray(weights)[order]
    positives = float((weight * positive).sum())
    negatives = float((weight * ~positive).sum())
    result = dict(tokens=float(weight.sum()), positive_tokens=positives,
                  negative_tokens=negatives, auroc=None, ap=None)
    if not positives:
        return result
    true_positive = np.cumsum(weight * positive)[ends]
    false_positive = np.cumsum(weight * ~positive)[ends]
    previous_true = np.r_[0., true_positive[:-1]]
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-300)
    result['ap'] = float(((true_positive - previous_true) * precision).sum() / positives)
    if negatives:
        false_increment = np.diff(np.r_[0., false_positive])
        area = (false_increment * (true_positive + previous_true) / 2).sum()
        result['auroc'] = float(area / (positives * negatives))
    return result


def binary_metrics(labels, scores):
    return rank_metrics(ranking(labels, scores))


def collect_tokens(answers, key, selection):
    selected = [(answer, selection(answer)) for answer in answers if key in answer['scores']]
    labels = np.concatenate([answer['labels'][mask] for answer, mask in selected])
    scores = np.concatenate([answer['scores'][key][mask] for answer, mask in selected])
    return labels, scores


def within_answer_metrics(answers, key):
    aucs, aps, answer_ids = [], [], []
    for answer in answers:
        if key not in answer['scores']:
            continue
        valid = answer['valid']
        metric = binary_metrics(answer['labels'][valid], answer['scores'][key][valid])
        if metric['auroc'] is not None:
            aucs.append(metric['auroc'])
            aps.append(metric['ap'])
            answer_ids.append(answer['record']['id'])
    return dict(answers_with_both_classes=len(aucs), response_ids=answer_ids,
                macro_auroc=float(np.mean(aucs)) if aucs else None,
                macro_ap=float(np.mean(aps)) if aps else None)


def normal_alarms(answers, key):
    normal = [answer for answer in answers if key in answer['scores']
              and not answer['labels'][answer['valid']].any()]
    token_count = sum(int(answer['valid'].sum()) for answer in normal)
    alarms = [answer['scores'][key][answer['valid']] > 0 for answer in normal]
    alarm_tokens = sum(int(values.sum()) for values in alarms)
    alarm_answers = sum(bool(values.any()) for values in alarms)
    return dict(threshold=0., rule='score > 0', normal_answers=len(normal),
                normal_tokens=token_count, alarm_tokens=alarm_tokens, alarm_answers=alarm_answers,
                token_fpr=alarm_tokens / token_count if token_count else None,
                any_alarm_rate=alarm_answers / len(normal) if normal else None)


def clean_prefix_mask(answer):
    errors = np.flatnonzero(answer['valid'] & answer['labels'])
    if not len(errors):
        return answer['valid']
    return answer['valid'] & (np.arange(len(answer['valid'])) <= errors[0])


def rival_coverage(answers, key, training_ids):
    rival_key = key.removesuffix('_gap') + '_rival_ids'
    rivals = np.concatenate([answer['scores'][rival_key][answer['valid']] for answer in answers])
    original = np.concatenate([answer['scores']['actual_ids'][answer['valid']] for answer in answers])
    trained_rival = np.isin(rivals, training_ids)
    outside = ~np.isin(original, training_ids)
    return dict(tokens=len(rivals), rival_in_training_vocabulary=int(trained_rival.sum()),
        rival_in_training_vocabulary_fraction=float(trained_rival.mean()) if len(rivals) else None,
        outside_training_original_tokens=int(outside.sum()),
        outside_training_original_with_training_rival=int((outside & trained_rival).sum()),
        outside_training_original_training_rival_fraction=float(trained_rival[outside].mean()) if outside.any() else None,
        interpretation='Vocabulary support only; rival semantic correctness is unsupported.')


def method_report(answers, key, training_ids):
    labels, scores = collect_tokens(answers, key, lambda answer: answer['valid'])
    result = dict(all_tokens=binary_metrics(labels, scores),
                  within_answer=within_answer_metrics(answers, key))
    masks = dict(
        greedy_equals_original=lambda answer: answer['valid'] &
            (answer['scores']['actual_ids'] == answer['scores']['greedy_ids']),
        training_vocabulary=lambda answer: answer['valid'] &
            np.isin(answer['scores']['actual_ids'], training_ids),
        outside_training_vocabulary=lambda answer: answer['valid'] &
            ~np.isin(answer['scores']['actual_ids'], training_ids),
        clean_prefix_through_first_error=clean_prefix_mask,
    )
    for name, select in masks.items():
        labels, scores = collect_tokens(answers, key, select)
        result[name] = binary_metrics(labels, scores)
    result['answers_available'] = sum(key in answer['scores'] for answer in answers)
    if key.endswith('_gap'):
        result['normal_answer_alarms'] = normal_alarms(answers, key)
        result['rival_vocabulary_coverage'] = rival_coverage(answers, key, training_ids)
    return result


def native_report(answers, key, training_ids):
    selected = lambda answer: answer['valid'] & (answer['scores']['actual_ids'] == answer['scores']['greedy_ids'])
    labels, scores = collect_tokens(answers, key, selected)
    return dict(greedy_equals_original=binary_metrics(labels, scores),
                rival_vocabulary_coverage=rival_coverage(answers, key, training_ids),
                interpretation='Original labels used only where observer greedy equals original token; no truth assigned to other greedy choices.')


def training_vocabulary(roster):
    path = Path(roster['input_paths']['source_program_controls'])
    controls = json.loads(path.read_text())
    return sorted({token for row in controls if row['cohort'] == 'fit' for token in row['candidate_ids']})


def coverage_report(answers, training_ids):
    valid = np.concatenate([answer['valid'] for answer in answers])
    actual = np.concatenate([answer['scores']['actual_ids'] for answer in answers])
    greedy = np.concatenate([answer['scores']['greedy_ids'] for answer in answers])
    topk = np.concatenate([answer['scores']['actual_in_topk'] for answer in answers])
    candidates = np.concatenate([answer['scores']['candidate_ids'] for answer in answers])
    labels = np.concatenate([answer['labels'] for answer in answers])
    training = np.isin(actual, training_ids)
    return dict(answers=len(answers), sources=len({a['record']['source_id'] for a in answers}),
        raw_tokens=len(valid), valid_tokens=int(valid.sum()), excluded_zero_width_or_special=int((~valid).sum()),
        positive_tokens=int((valid & labels).sum()), normal_answers=sum(not a['labels'][a['valid']].any() for a in answers),
        error_answers=sum(bool(a['labels'][a['valid']].any()) for a in answers),
        training_token_ids=training_ids, training_token_count=len(training_ids),
        training_actual_tokens=int((valid & training).sum()), outside_training_actual_tokens=int((valid & ~training).sum()),
        actual_in_top32=int((valid & topk).sum()), actual_top32_fraction=float(topk[valid].mean()),
        greedy_equals_original_tokens=int((valid & (actual == greedy)).sum()),
        greedy_equals_original_fraction=float((actual[valid] == greedy[valid]).mean()),
        top32_contains_training_token=int(np.isin(candidates[valid], training_ids).any(axis=1).sum()),
        semantic_candidate_accuracy='unsupported: no semantic rival labels are provided')


def onset_reports(answers):
    records = []
    for answer in answers:
        onsets = np.flatnonzero(answer['valid'] & answer['onsets'])
        if not len(onsets):
            continue
        first = int(onsets[0])
        valid = answer['valid']
        positions = np.arange(len(valid))
        previous = valid & (positions < first)
        scores = {}
        for key in score_keys(OBSERVED_SUFFIXES):
            alarms = valid & (answer['scores'][key] > 0)
            indices = np.flatnonzero(alarms)
            scores[key] = dict(score=float(answer['scores'][key][first]), alarm=bool(alarms[first]),
                earlier_normal_prefix_alarm_tokens=int((alarms & previous).sum()),
                first_alarm_position=int(indices[0]) if len(indices) else None,
                first_alarm_equals_first_error=bool(len(indices) and indices[0] == first))
        native = {key: float(answer['scores'][key][first]) for key in score_keys(('native_gap',))}
        records.append(dict(response_id=answer['record']['id'], source_id=answer['record']['source_id'],
            first_onset=first, labelled_span_onsets=onsets.tolist(),
            actual_id=int(answer['scores']['actual_ids'][first]),
            actual_in_top32=bool(answer['scores']['actual_in_topk'][first]),
            greedy_equals_original=bool(answer['scores']['greedy_ids'][first] == answer['scores']['actual_ids'][first]),
            clean_prefix_normal_tokens=int(previous.sum()), observed_scores=scores, native_greedy_scores=native,
            interpretation='Pre-acceptance observer state at first annotated error; earlier normal-prefix alarms are false alarms, not prevention successes.'))
    return records


def source_bootstrap(answers, suffix, draws, seed):
    """Paired source-block resampling; average three seed metric differences."""
    sources = sorted({answer['record']['source_id'] for answer in answers})
    source_index = {source: index for index, source in enumerate(sources)}
    labels = np.concatenate([answer['labels'][answer['valid']] for answer in answers])
    token_sources = np.concatenate([np.full(int(answer['valid'].sum()), source_index[answer['record']['source_id']]) for answer in answers])
    ranked = {}
    for variant in ('ordered', 'mean'):
        for model_seed in SEEDS:
            key = f'{variant}_seed{model_seed}_{suffix}'
            values = np.concatenate([answer['scores'][key][answer['valid']] for answer in answers])
            ranked[variant, model_seed] = ranking(labels, values)
    generator = np.random.default_rng(seed)
    sampled = {metric: {str(s): [] for s in SEEDS} for metric in ('auroc', 'ap')}
    for values in sampled.values():
        values['mean_of_seed_deltas'] = []
    for _ in range(draws):
        multiplicity = np.bincount(generator.integers(len(sources), size=len(sources)), minlength=len(sources))
        metrics = {name: rank_metrics(curve, multiplicity[token_sources]) for name, curve in ranked.items()}
        for metric, values in sampled.items():
            pairs = [(metrics['ordered', s][metric], metrics['mean', s][metric]) for s in SEEDS]
            if all(left is not None and right is not None for left, right in pairs):
                differences = [left - right for left, right in pairs]
                for model_seed, difference in zip(SEEDS, differences):
                    values[str(model_seed)].append(float(difference))
                values['mean_of_seed_deltas'].append(float(np.mean(differences)))
    native = {name: rank_metrics(curve) for name, curve in ranked.items()}
    return bootstrap_report(native, sampled, suffix, draws, seed, sources)


def bootstrap_report(native, sampled, suffix, draws, seed, sources):
    result = dict(suffix=suffix, draws=draws, bootstrap_seed=seed, sources=sources,
                  unit='source; all six answers/tokens retained together',
                  statistic='primary seed42 paired pooled metric difference; other seeds and their mean are fixed replications; seed uncertainty not bootstrapped')
    for metric, samples in sampled.items():
        pairs = [(native['ordered', s][metric], native['mean', s][metric]) for s in SEEDS]
        point = float(np.mean([left - right for left, right in pairs])) if all(left is not None and right is not None for left, right in pairs) else None
        per_seed = {}
        for model_seed in SEEDS:
            left, right = native['ordered', model_seed][metric], native['mean', model_seed][metric]
            values = samples[str(model_seed)]
            per_seed[str(model_seed)] = dict(ordered_minus_mean=left - right if left is not None and right is not None else None,
                ci95=np.quantile(values, [.025, .975]).tolist() if values else None,
                defined_draws=len(values), undefined_draws=draws - len(values))
        values = samples['mean_of_seed_deltas']
        result[metric] = dict(primary_seed42=per_seed['42'], fixed_replication_seeds=per_seed,
            mean_of_seed_deltas=dict(ordered_minus_mean=point,
                ci95=np.quantile(values, [.025, .975]).tolist() if values else None,
                defined_draws=len(values), undefined_draws=draws - len(values)))
    return result


def evaluate(root, roster_path, *, draws=2000, seed=73):
    freeze, roster, verified = verify_freeze(root, roster_path)
    verify_evaluation_protocol(root, freeze, verified, draws, seed)
    answers = load_answers(root, roster, verified)
    training_ids = training_vocabulary(roster)
    if len(training_ids) != 6:
        raise ValueError('B1 requires the frozen six-token source-program vocabulary')
    # The first annotation-file open occurs only after every check above passes.
    load_annotations(roster, answers)
    keys = score_keys(OBSERVED_SUFFIXES) + [key for key in BASELINES if any(key in answer['scores'] for answer in answers)]
    methods = {key: method_report(answers, key, training_ids) for key in keys}
    native = {key: native_report(answers, key, training_ids) for key in score_keys(('native_gap',))}
    generators = sorted({answer['record']['generator'] for answer in answers})
    by_generator = {name: {key: method_report([a for a in answers if a['record']['generator'] == name], key, training_ids)
                          for key in keys if any(a['record']['generator'] == name and key in a['scores'] for a in answers)} for name in generators}
    return dict(status='DONE', schema='source_uptake_evaluation_v1',
        scope='exploratory/current-method-source-disjoint original-answer observer evaluation; historical corpus exposed',
        primary_method='ordered_seed42_gap', fixed_replication_seeds=[123, 2026],
        primary_key='ordered_seed42_gap',
        secondary_score='g + native logp, fixed coefficient 1; *_logp_gap',
        source_derived_self_supervised=True, natural_labels_used_for_fit=False, same_generator_prevention=False,
        observed_token_interpretation='y_t is a candidate embedding; native predictor excludes y_t. This is before acceptance of an observed original token, not prediction of a new generator token truth.',
        freeze_sha256=sha256(root / 'FREEZE.json'), roster_sha256=freeze['roster']['sha256'],
        evaluation_protocol_sha256=sha256(root / 'EVALUATION_PROTOCOL.json'),
        coverage=coverage_report(answers, training_ids), methods=methods, by_generator=by_generator,
        native_greedy_matched_only=native, first_onsets=onset_reports(answers),
        ordered_minus_mean={suffix: source_bootstrap(answers, suffix, draws, seed) for suffix in OBSERVED_SUFFIXES},
        prior_baseline_scope='Historical unit-broadcast/post-token scalar comparators; not causal pre-token baselines with equal input features, and not new localization.',
        prior_baselines_unavailable=[key for key in PRIOR_KEYS if key not in methods],
        threshold_selection='none; gap > 0 is an uncalibrated diagnostic alarm, not a controlled normal FPR guarantee',
        annotation_file_sha256=sha256(Path(roster['input_paths']['annotation_file_for_later_frozen_evaluation'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    args = parser.parse_args()
    output = args.output or args.root / 'EVALUATION.json'
    if output.exists():
        raise ValueError(f'Refusing to overwrite evaluation: {output}')
    if args.bootstrap_draws < 1:
        raise ValueError('Source bootstrap requires at least one draw')
    result = evaluate(args.root, args.roster, draws=args.bootstrap_draws)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), coverage=result['coverage'],
                          ordered_minus_mean=result['ordered_minus_mean']), indent=2), flush=True)


if __name__ == '__main__':
    main()
