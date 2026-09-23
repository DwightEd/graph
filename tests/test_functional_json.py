"""Malformed model-output recovery and the one-command audit launcher."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from state_audit.storage import read_json, write_json
from experiments.native_support.functional_bank import (
    InvalidCandidateBank, VALIDATION, cached_generation, parse_generated_json, prepare_bank,
)
from experiments.native_support.functional_run import arguments, execute


BROKEN_VERDICT = '{"valid": false, "reason": "unfinished'
PROPOSAL = dict(paraphrase='Rewrite.', polarity=[], binding=[])


class BankTokenizer:
    def decode(self, ids, **kwargs):
        return ''.join({1: 'Question', 5: 'Answer.'}[value] for value in ids)

    def encode(self, text, **kwargs):
        return {'Rewrite.': [6]}[text]


@pytest.fixture
def response():
    return dict(id='a', source_id='s', prompt_length=1,
                token_ids=[1, 5], token_text=['Question', 'Answer.'])


@pytest.mark.parametrize('text', [
    '{"valid": false, "reason": "different meanings"}',
    '```json\n{"valid": false, "reason": "different meanings"}\n```',
])
def test_complete_json_preserves_false_verdict(text):
    assert parse_generated_json(text) == dict(valid=False, reason='different meanings')


@pytest.mark.parametrize('text', [BROKEN_VERDICT, '{"valid": true}\nextra commentary'])
def test_incomplete_or_extra_content_is_not_salvaged(text):
    with pytest.raises(json.JSONDecodeError):
        parse_generated_json(text)


@pytest.mark.parametrize('broken', [BROKEN_VERDICT, '{"valid": "false", "reason": "wrong type"}'])
def test_saved_bad_output_retried_once_without_overwriting_or_reinterpreting(tmp_path, broken):
    path = tmp_path / 'validation.json'
    write_json(path, {'raw_output': broken})
    original = path.read_bytes()
    verdict = dict(valid=False, reason='different meanings')
    def generate(model, tokenizer, instruction, payload, destination, budget):
        assert destination.name == 'validation_json_retry.json'
        assert 'json_format_feedback' in payload
        write_json(destination, {'raw_output': json.dumps(verdict)})
        return verdict
    with patch('experiments.native_support.functional_bank.generate_json', side_effect=generate) as generation:
        result = cached_generation(None, None, VALIDATION, {}, path, 100, kind='validation')
    assert result == verdict and generation.call_count == 1
    assert path.read_bytes() == original
    with patch('experiments.native_support.functional_bank.generate_json', side_effect=AssertionError('regenerated')):
        assert cached_generation(None, None, VALIDATION, {}, path, 100, kind='validation') == verdict


def test_exhausted_json_retry_is_rejected_and_cached(tmp_path, response):
    write_json(tmp_path / 'proposal.json', {'raw_output': json.dumps(PROPOSAL)})
    for name in ('validation.json', 'validation_json_retry.json'):
        write_json(tmp_path / name, {'raw_output': BROKEN_VERDICT})
    with patch('experiments.native_support.functional_bank.generate_json', side_effect=AssertionError('regenerated')):
        bank = prepare_bank(None, BankTokenizer(), response, 0, 1, tmp_path, 100)
        assert prepare_bank(None, BankTokenizer(), response, 0, 1, tmp_path, 100) == bank
    assert bank['valid'] is False and bank['reason'] == 'invalid_generated_json'
    assert bank['generation_error']['file'] == 'validation_json_retry.json'
    assert not list(tmp_path.glob('candidate_*.npz'))
    assert read_json(tmp_path / 'validation.json')['raw_output'] == BROKEN_VERDICT


def test_fresh_truncated_proposal_has_bounded_format_retry(tmp_path, response):
    def generate(model, tokenizer, instruction, payload, destination, budget):
        assert destination.name in ('proposal.json', 'proposal_json_retry.json')
        raw = '{"paraphrase": "unfinished'
        write_json(destination, {'raw_output': raw})
        return parse_generated_json(raw)
    with patch('experiments.native_support.functional_bank.generate_json', side_effect=generate) as generation:
        bank = prepare_bank(None, BankTokenizer(), response, 0, 1, tmp_path, 100)
    assert generation.call_count == 2
    assert bank['reason'] == 'invalid_generated_json' and bank['valid'] is False


def test_generation_runtime_failure_is_not_converted_to_rejection(tmp_path):
    with patch('experiments.native_support.functional_bank.generate_json', side_effect=RuntimeError('model failure')), \
         pytest.raises(RuntimeError, match='model failure'):
        cached_generation(None, None, VALIDATION, {}, tmp_path / 'validation.json', 100, kind='validation')


def test_failed_json_unit_does_not_block_later_capture(tmp_path, response):
    rejected, accepted = tmp_path / 'first', tmp_path / 'second'
    write_json(rejected / 'proposal.json', {'raw_output': json.dumps(PROPOSAL)})
    for name in ('validation.json', 'validation_json_retry.json'):
        write_json(rejected / name, {'raw_output': BROKEN_VERDICT})
    def generate(model, tokenizer, instruction, payload, destination, budget):
        assert destination.parent == accepted
        result = PROPOSAL if destination.name == 'proposal.json' else dict(valid=True, reason='fixture')
        write_json(destination, {'raw_output': json.dumps(result)})
        return result
    args = SimpleNamespace(stage='run', device='cpu', dtype='float32', max_new_tokens=100)
    identity = dict(start=0, stop=1)
    jobs = [(response, {}, directory, identity) for directory in (rejected, accepted)]
    with patch('state_audit.model.load_model', return_value=(object(), BankTokenizer())), \
         patch('experiments.native_support.functional_bank.generate_json', side_effect=generate), \
         patch('experiments.native_support.functional_run.capture_bank') as capture:
        execute(args, dict(model='fixture'), jobs)
    assert capture.call_count == 1
    assert capture.call_args.args[4] == accepted
    assert read_json(rejected / 'bank.json')['reason'] == 'invalid_generated_json'
    assert read_json(accepted / 'bank.json')['valid'] is True


def test_launcher_resumes_one_process_forwards_arguments_and_preserves_exit_code(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / 'experiments/native_support/run_functions.sh'
    executable = tmp_path / 'python fixture'
    executable.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                          'print(json.dumps(dict(cwd=os.getcwd(), args=sys.argv[1:])))\n'
                          'raise SystemExit(23)\n')
    executable.chmod(0o755)
    output = tmp_path / 'result with spaces'
    result = subprocess.run(['bash', str(launcher), '--output', str(output), '--device', 'cpu'],
                            cwd=tmp_path, env={**os.environ, 'FUNCTION_AUDIT_PYTHON': str(executable)},
                            text=True, capture_output=True)
    assert result.returncode == 23, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation['cwd'] == str(root)
    assert invocation['args'][:3] == ['-u', 'main.py', 'transport-functions']
    parsed = arguments(invocation['args'][3:])
    assert parsed.stage == 'run' and parsed.resume is True
    assert parsed.output == output and parsed.device == 'cpu'
    assert parsed.response_ids == ['12219']
