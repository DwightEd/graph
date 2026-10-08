"""Freeze-before-label access and all-token evaluation scientific contracts."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import evaluate as evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_run(root):
    records, inputs, annotation_rows = [], {}, []
    for identity, actual, text, offsets, labels, values in [
        ('a', [1, 2, 7, 4], 'a b c d', [[0, 1], [2, 3], [4, 5], [6, 7]],
         [{'start': 4, 'end': 5, 'text': 'c'}], [-1., -.5, 2., -.2]),
        ('b', [1, 2, 3, 4], 'a b c d', [[0, 1], [2, 3], [4, 5], [6, 7]],
         [], [0., -1., .25, 0.]),
        ('c', [99, 1, 2, 3], 'abc', [[0, 0], [0, 1], [1, 2], [2, 3]],
         [{'start': 0, 'end': 1, 'text': 'a'}], [8., 2., -1., -1.]),
    ]:
        source = root / 'prepared' / f'{identity}_source.json'
        response = root / 'prepared' / f'{identity}_response.json'
        write_json(source, dict(prompt_with_source=[20, 21], source_mask=[True, True]))
        write_json(response, dict(answer_ids=actual, text=text, offsets=offsets, special_ids=[99]))
        hashes = {str(path): evaluation.sha256(path) for path in (source, response)}
        inputs.update(hashes)
        records.append(dict(id=identity, source_id=identity, tokens=4, generator='fixture',
                            source_cache_path=str(source), response_cache_path=str(response)))
        directory = root / 'answers' / identity
        write_json(directory / 'data.json', dict(response_id=identity, source_id=identity,
            text=text, offsets=offsets, special_ids=[99], input_hashes=hashes))
        actual_ids = np.array(actual)
        greedy = actual_ids.copy()
        if identity == 'a':
            greedy[0] = 55
        candidates = np.column_stack([greedy] + [np.full(4, token) for token in range(100, 131)])
        scores = dict(actual_ids=actual_ids, greedy_ids=greedy, candidate_ids=candidates,
            actual_in_topk=(candidates == actual_ids[:, None]).any(1),
            entropy=np.array([.1, .2, .3, .4]),
            candidate_logp=np.tile(np.linspace(-3.6, -6.7, 32), (4, 1)))
        scores['actual_logp'] = np.where(scores['actual_in_topk'], -3.6, -9.)
        scores['surprisal'] = -scores['actual_logp']
        for key in evaluation.score_keys(('gap', 'logp_gap', 'native_gap')):
            scores[key] = np.array(values)
            scores[key.removesuffix('_gap') + '_rival_ids'] = candidates[:, 1]
        for key in evaluation.PRIOR_KEYS:
            scores[key] = np.array(values)
        # Semantic rival IDs are metadata; they must never become detection scores.
        np.savez(directory / 'scores.npz', **scores)
        annotation_rows.append(dict(id=identity, source_id=identity, response=text, labels=labels))

    controls = root / 'program.json'
    write_json(controls, [dict(cohort='fit', candidate_ids=[n, n + 1]) for n in (1, 3, 5)])
    inputs[str(controls)] = evaluation.sha256(controls)
    annotations = root / 'annotations.jsonl'
    with annotations.open('w') as stream:
        for row in annotation_rows:
            stream.write(json.dumps(row) + '\n')
        # Non-roster labels are intentionally unusable; the row must not be parsed.
        stream.write('{"id": "not-selected", "labels": INVALID}\n')
    roster = root / 'roster.json'
    write_json(roster, dict(natural=dict(records=records, answers=3, tokens=12), input_sha256=inputs,
        input_paths=dict(source_program_controls=str(controls),
                         annotation_file_for_later_frozen_evaluation=str(annotations))))
    files = {str(path.relative_to(root)): evaluation.sha256(path)
             for path in root.glob('answers/*/*')}
    checkpoints = {}
    for variant in evaluation.VARIANTS:
        for seed in evaluation.SEEDS:
            path = root / 'checkpoints' / f'{variant}_seed{seed}' / 'model.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'fixture-checkpoint')
            checkpoints[str(path)] = evaluation.sha256(path)
    metric_hashes = {str(path): evaluation.sha256(path) for path in evaluation.metric_code_paths()}
    protocol = root / 'EVALUATION_PROTOCOL.json'
    write_json(protocol, dict(primary_key='ordered_seed42_gap', fixed_replication_seeds=[123, 2026],
        threshold=0, bootstrap_draws=32, bootstrap_seed=73, metric_code_hashes=metric_hashes))
    files[str(protocol)] = evaluation.sha256(protocol)
    files.update(metric_hashes)
    write_json(root / 'FREEZE.json', dict(status='COMPLETE', files=files,
        roster=dict(path=str(roster), sha256=evaluation.sha256(roster)), checkpoints=checkpoints))
    return roster


class EvaluationContracts(unittest.TestCase):
    def test_tied_metrics_and_source_multiplicity(self):
        labels = np.array([False, True, False, True])
        scores = np.array([0., 0., 1., 2.])
        result = evaluation.binary_metrics(labels, scores)
        self.assertAlmostEqual(result['auroc'], .625)
        self.assertAlmostEqual(result['ap'], .75)
        weighted = evaluation.rank_metrics(evaluation.ranking(labels, scores), [1, 3, 2, 1])
        self.assertAlmostEqual(weighted['auroc'], .375)
        self.assertAlmostEqual(weighted['ap'], 19 / 28)

    def test_tampered_score_blocks_annotation_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            with (root / 'answers/a/scores.npz').open('ab') as stream:
                stream.write(b'tampered')
            with mock.patch.object(evaluation, 'load_annotations') as reader:
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    evaluation.evaluate(root, roster, draws=8)
                reader.assert_not_called()

    def test_missing_answer_freeze_blocks_annotation_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            freeze = json.loads((root / 'FREEZE.json').read_text())
            del freeze['files']['answers/b/scores.npz']
            write_json(root / 'FREEZE.json', freeze)
            with mock.patch.object(evaluation, 'load_annotations') as reader:
                with self.assertRaisesRegex(ValueError, 'absent from freeze'):
                    evaluation.evaluate(root, roster, draws=8)
                reader.assert_not_called()

    def test_annotation_hash_is_not_an_early_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            freeze = json.loads((root / 'FREEZE.json').read_text())
            freeze['files']['annotations.jsonl'] = 'must-not-be-opened-for-hashing'
            write_json(root / 'FREEZE.json', freeze)
            with mock.patch.object(evaluation, 'load_annotations') as reader:
                with self.assertRaisesRegex(ValueError, 'pre-evaluation input hashing'):
                    evaluation.evaluate(root, roster, draws=8)
                reader.assert_not_called()

    def test_invalid_native_rival_blocks_annotation_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            path = root / 'answers/a/scores.npz'
            with np.load(path) as saved:
                scores = {key: saved[key] for key in saved.files}
            scores['ordered_seed42_native_rival_ids'][0] = scores['greedy_ids'][0]
            np.savez(path, **scores)
            freeze = json.loads((root / 'FREEZE.json').read_text())
            freeze['files']['answers/a/scores.npz'] = evaluation.sha256(path)
            write_json(root / 'FREEZE.json', freeze)
            with mock.patch.object(evaluation, 'load_annotations') as reader:
                with self.assertRaisesRegex(ValueError, 'distinct native top32 candidate'):
                    evaluation.evaluate(root, roster, draws=32)
                reader.assert_not_called()

    def test_frozen_evaluation_configuration_blocks_changes_before_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            with mock.patch.object(evaluation, 'load_annotations') as reader:
                with self.assertRaisesRegex(ValueError, 'protocol mismatch: bootstrap_draws'):
                    evaluation.evaluate(root, roster, draws=64)
                reader.assert_not_called()

    def test_exact_alignment_all_token_coverage_and_normal_alarms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roster = synthetic_run(root)
            result = evaluation.evaluate(root, roster, draws=32)
            self.assertEqual(result['coverage']['raw_tokens'], 12)
            self.assertEqual(result['coverage']['valid_tokens'], 11)
            self.assertEqual(result['coverage']['positive_tokens'], 2)
            self.assertEqual(result['coverage']['outside_training_actual_tokens'], 1)
            method = result['methods']['ordered_seed42_gap']
            alarms = method['normal_answer_alarms']
            self.assertEqual(alarms['normal_answers'], 1)
            self.assertEqual(alarms['alarm_tokens'], 1)
            self.assertEqual(alarms['token_fpr'], .25)
            self.assertEqual(alarms['any_alarm_rate'], 1.)
            onsets = {row['response_id']: row for row in result['first_onsets']}
            self.assertEqual(onsets['a']['first_onset'], 2)
            self.assertEqual(onsets['c']['first_onset'], 1)
            self.assertNotIn('ordered_seed42_rival_ids', result['methods'])
            self.assertNotIn('ordered_seed42_native_gap', result['methods'])
            self.assertEqual(result['native_greedy_matched_only']['ordered_seed42_native_gap']
                             ['greedy_equals_original']['tokens'], 10)
            json.dumps(result, allow_nan=False)

    def test_bootstrap_keeps_source_blocks_and_reports_primary_seed(self):
        answers = []
        for source in ('source-a', 'source-b'):
            scores = {}
            for seed in evaluation.SEEDS:
                scores[f'ordered_seed{seed}_gap'] = np.array([-1., 1.])
                scores[f'mean_seed{seed}_gap'] = np.array([1., -1.])
            answers.append(dict(record=dict(source_id=source), valid=np.ones(2, bool),
                                labels=np.array([False, True]), scores=scores))
        result = evaluation.source_bootstrap(answers, 'gap', 64, 73)
        self.assertEqual(result['auroc']['primary_seed42']['ci95'], [1., 1.])
        self.assertEqual(result['ap']['primary_seed42']['ci95'], [.5, .5])
        self.assertEqual(result['auroc']['primary_seed42']['undefined_draws'], 0)


if __name__ == '__main__':
    unittest.main()
