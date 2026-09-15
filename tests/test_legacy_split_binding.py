"""Legacy six-field caches: partitions come from input folders, not output names."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from experiments.unsupervised_token_graph.evaluation_data import (
    EvaluationBinding, prepare_record, read_sources, resolve_tokenizer, verified_offsets,
)
from experiments.unsupervised_token_graph.reanchor_evaluate import evaluate


class CharacterTokenizer:
    all_special_ids = [1, 2]

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, return_offsets_mapping=True)
        return dict(input_ids=[ord(c) + 10 for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return ''.join(chr(i - 10) for i in ids if i not in self.all_special_ids)


def fake_transformers(monkeypatch):
    calls = []
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert kwargs == dict(use_fast=True, local_files_only=True)
            calls.append(path)
            return CharacterTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    return calls


def legacy_run(tmp_path, partition='train', with_alignment=True):
    cache = tmp_path / 'RAGTruth/attention/llama31_8b' / partition
    cache.mkdir(parents=True)
    root = tmp_path / 'outputs/unrelated_output_name'
    root.mkdir(parents=True)
    settings = dict(version='source-carrier-information-v1', labels_read=False, cache=str(cache))
    (root / 'settings.json').write_text(json.dumps(settings))
    dataset = tmp_path / 'RAGTruth/dataset'; dataset.mkdir()
    gold = []
    response = 'abc'
    for rid in ('100', '101'):
        record = dict(id='attention_' + rid, source_id='', split='', task='', generator='',
                      cache='attention/attention_' + rid + '.npz',
                      file='samples/attention/attention_' + rid + '.npz', response_sha256='')
        arrays = dict(token_ids=[1, 50, 107, 108, 109], prompt_length=2, total_tokens=5,
                      query_positions=[2, 3, 4], prediction_positions=[3, 4, 5], cache_format='canonical_csr',
                      layer_ids=[0], head_ids=[0], source_mismatch_bits=[[np.nan, .8, .2]],
                      event_strength=[[.8, .2, .1]], permuted_source_mismatch_bits=[[np.nan, .3, .7]],
                      prompt_reach=[[.7, .6, .5]], unknown_mass=[[.1, .2, .1]])
        if with_alignment:
            record['response_sha256'] = hashlib.sha256(response.encode()).hexdigest()
            arrays['offsets'] = [[0, 1], [1, 2], [2, 3]]
        file = root / record['file']; file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(file, record_json=json.dumps(record), **arrays)
        gold.append(dict(id=rid, source_id='s' + rid, split=partition, response=response, model='fixture_generator',
                         labels=[dict(start=1, end=2)] if rid == '100' else []))
    annotations = dataset / 'response.jsonl'
    annotations.write_text('\n'.join(map(json.dumps, gold)))
    (dataset / 'source_info.json').write_text(json.dumps([dict(source_id='s100', task_type='QA'), dict(source_id='s101', task_type='Summary')]))
    return root, annotations, cache


def report_for(root, annotations, **kwargs):
    return evaluate(root, annotations, root / 'evaluation_partial.json', split='train', bootstrap=0, completed_only=True, **kwargs)


def snapshot(root):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file() and p.name != 'evaluation_partial.json'}


def test_missing_split_uses_input_folder_and_official_annotation(tmp_path):
    root, annotations, cache = legacy_run(tmp_path)
    before = snapshot(root)
    report = report_for(root, annotations)
    assert report['input_cache'] == str(cache)
    assert report['evaluated_responses'] == 2
    assert {r['split'] for r in report['evaluated_records']} == {'train'}
    assert {r['id'] for r in report['evaluated_records']} == {'100', '101'}
    assert {'ALL', 'QA|fixture_generator', 'Summary|fixture_generator'} == set(report['groups'])
    assert report['identity_binding'][0]['split_origin'] == 'input_cache_directory'
    assert before == snapshot(root)
    assert not (root / 'complete.json').exists()


def test_request_does_not_relabel_train_as_test(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    with pytest.raises(ValueError, match='saved splits=.*train'):
        evaluate(root, annotations, split='test', completed_only=True)


def test_test_directory_uses_only_test_samples(tmp_path):
    root, annotations, _ = legacy_run(tmp_path, partition='test')
    report = evaluate(root, annotations, split='test', bootstrap=0, completed_only=True)
    assert {r['split'] for r in report['evaluated_records']} == {'test'}


def test_six_field_result_can_recover_alignment_without_rescoring(tmp_path, monkeypatch):
    root, annotations, _ = legacy_run(tmp_path, with_alignment=False)
    calls = fake_transformers(monkeypatch)
    before = snapshot(root)
    report = report_for(root, annotations, tokenizer='/declared/original/tokenizer')
    assert calls == ['/declared/original/tokenizer']
    assert report['groups']['ALL']['views']['all_error']['event_strength']['evaluated_tokens'] == 4
    assert report['identity_binding'][0]['alignment_origin'] == 'exact_token_id_verified_offsets'
    assert snapshot(root) == before


def test_missing_offsets_are_reported_without_fabrication(tmp_path):
    root, annotations, _ = legacy_run(tmp_path, with_alignment=False)
    with pytest.raises(ValueError, match='TOKENIZER.*do not rerun'):
        report_for(root, annotations)


@pytest.mark.parametrize('field,value,message', [
    ('split', 'test', 'split mismatch'),
    ('source_id', 'wrong', 'identity mismatch'),
    ('response', 'abd', 'identity mismatch'),
])
def test_existing_identity_checks_are_not_weakened(tmp_path, field, value, message):
    root, annotations, _ = legacy_run(tmp_path)
    rows = [json.loads(s) for s in annotations.read_text().splitlines()]
    if field == 'source_id':
        # This field must be present in the saved record to constrain the join.
        file = next((root / 'samples').rglob('*.npz'))
        with np.load(file) as data:
            arrays = {k: data[k] for k in data.files}
        record = json.loads(str(arrays['record_json'])); record['source_id'] = 's100'
        arrays['record_json'] = json.dumps(record); np.savez_compressed(file, **arrays)
    rows[0][field] = value
    annotations.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match=message):
        report_for(root, annotations)


def test_wrong_token_ids_rejected_even_with_same_length(tmp_path, monkeypatch):
    root, annotations, _ = legacy_run(tmp_path, with_alignment=False)
    fake_transformers(monkeypatch)
    rows = [json.loads(s) for s in annotations.read_text().splitlines()]; rows[0]['response'] = 'abd'
    annotations.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='token_ids do not match'):
        report_for(root, annotations, tokenizer='/original')


def test_manifest_resolves_existing_tokenizer_without_new_population(tmp_path, monkeypatch):
    root, annotations, cache = legacy_run(tmp_path, with_alignment=False)
    tokenizer = tmp_path / 'existing-model'; tokenizer.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'model_path': str(tokenizer)}))
    calls = fake_transformers(monkeypatch)
    report_for(root, annotations)
    assert calls == [str(tokenizer.resolve())]


def test_report_directory_name_cannot_assign_split(tmp_path):
    settings = {'cache': str(tmp_path / 'attention')}
    record = dict(id='1', split='', file='samples/train/1.npz', cache='1.npz')
    assert prepare_record(settings, record)['split'] == ''


def test_saved_split_conflicting_with_input_directory_is_not_overwritten():
    settings = {'cache': '/RAGTruth/attention/test'}
    with pytest.raises(ValueError, match='conflicts'):
        prepare_record(settings, dict(id='1', split='train', cache='1.npz', file='samples/1.npz'))


def test_no_folder_split_can_use_official_record_but_not_cli_label(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    settings = json.loads((root / 'settings.json').read_text()); settings['cache'] = str(tmp_path / 'attention')
    (root / 'settings.json').write_text(json.dumps(settings))
    report = report_for(root, annotations)
    assert all(r['split_origin'] == 'annotation' for r in report['identity_binding'])


def test_saved_offsets_and_hash_do_not_require_transformers(tmp_path, monkeypatch):
    root, annotations, _ = legacy_run(tmp_path)
    monkeypatch.setitem(sys.modules, 'transformers', None)
    report_for(root, annotations)


def test_partial_and_full_same_cohort_have_same_metrics(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    partial = report_for(root, annotations)
    records = []
    for file in sorted((root / 'samples').rglob('*.npz')):
        with np.load(file) as data:
            records.append(json.loads(str(data['record_json'])))
    (root / 'complete.json').write_text(json.dumps(dict(complete=True, responses=2)))
    (root / 'summary.json').write_text(json.dumps(dict(version='source-carrier-information-v1', labels_read=False, responses=records)))
    full = evaluate(root, annotations, split='train', bootstrap=0)
    assert partial['groups'] == full['groups']


@pytest.mark.parametrize('form', ['list', 'map', 'jsonl', 'single', 'jsonl_in_json'])
def test_existing_source_info_layouts(tmp_path, form):
    rows = [dict(source_id='s1', task_type='Data2txt', source_info={'value': 1}),
            dict(source_id='s2', task_type='QA', source_info={'passages': 'evidence'})]
    path = tmp_path / ('source_info.jsonl' if form == 'jsonl' else 'source_info.json')
    if form in ('jsonl', 'jsonl_in_json'):
        text = '\n'.join(map(json.dumps, rows)) + '\n'
    else:
        value = rows if form == 'list' else ({r['source_id']: r for r in rows} if form == 'map' else rows[0])
        text = json.dumps(value)
    path.write_text(text, encoding='utf-8')
    sources, used = read_sources(tmp_path / 'response.jsonl')
    expected = rows[:1] if form == 'single' else rows
    assert sources == {r['source_id']: r for r in expected}
    assert used == str(path)


def test_source_info_joins_on_source_id_not_response_id(tmp_path):
    # Source records may contain an unrelated id; the join is always source_id.
    row = dict(id='not_the_source_key', source_id=123, task_type='QA')
    path = tmp_path / 'source_info.json'
    path.write_text(json.dumps([row]), encoding='utf-8')
    sources, _ = read_sources(tmp_path / 'response.jsonl')
    assert sources == {'123': row}


def test_offsets_handle_unicode_and_known_eos():
    tokenizer = CharacterTokenizer(); response = 'a中😀'
    ids = [1, 50] + tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)['input_ids'] + [2]
    np.testing.assert_array_equal(verified_offsets(tokenizer, ids, 2, response), [[0, 1], [1, 2], [2, 3], [3, 3]])


def test_offsets_reject_extra_nonspecial_tokens():
    with pytest.raises(ValueError, match='token_ids do not match'):
        verified_offsets(CharacterTokenizer(), [1, 50, 107, 108, 109, 999], 2, 'abc')


def test_context_tokenization_path_is_exact():
    class ContextTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text == 'abc':
                result['input_ids'] = [999, 108, 109]
            return result
    np.testing.assert_array_equal(verified_offsets(ContextTokenizer(), [50, 107, 108, 109], 1, 'abc'), [[0, 1], [1, 2], [2, 3]])


def test_generator_name_is_not_a_tokenizer_guess():
    assert resolve_tokenizer({'generator': 'llama31_8b'}, {'file': 'samples/1.npz'}) is None


def test_duplicate_annotation_id_is_not_silently_overwritten(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    text = annotations.read_text(); annotations.write_text(text + '\n' + text.splitlines()[0])
    with pytest.raises(ValueError, match='duplicate response ID'):
        report_for(root, annotations)


def test_cli_evaluates_saved_files_without_loading_scoring(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'experiments.unsupervised_token_graph.run', 'evaluate',
        '--predictions', str(root), '--annotations', str(annotations), '--output', str(root / 'evaluation_partial.json'),
        '--completed-only', '--split', 'train', '--bootstrap', '0'], cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'completed_samples_preview' in result.stdout
    assert 'input_cache' in result.stdout


def test_shell_reads_split_folders_and_passes_optional_existing_paths(tmp_path):
    root, annotations, _ = legacy_run(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PY=sys.executable, SPLIT='train', OUTPUT=str(root), ANNOTATIONS=str(annotations),
               TOKENIZER='/not_loaded_when_offsets_and_hash_exist', SOURCE_INFO=str(annotations.parent / 'source_info.json'),
               BOOTSTRAP='0')
    result = subprocess.run(['bash', 'experiments/unsupervised_token_graph/evaluate.sh', '--completed-only'],
                            cwd=repo, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((root / 'evaluation_partial.json').read_text())
    assert report['evaluated_responses'] == 2
    assert not (root / 'complete.json').exists()


def test_partial_snapshot_is_taken_before_annotations(tmp_path, monkeypatch):
    root, annotations, _ = legacy_run(tmp_path)
    original = Path.open
    def opening(path, *args, **kwargs):
        if path == annotations:
            other = root / 'samples/new.partial'; other.write_bytes(b'incomplete')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', opening)
    assert report_for(root, annotations)['completed_samples_found'] == 2


def test_no_identity_index_or_tokenizer_is_assumed_from_id_alone(tmp_path):
    root, annotations, _ = legacy_run(tmp_path, with_alignment=False)
    with pytest.raises(ValueError, match='identity-bound alignment'):
        report_for(root, annotations)
