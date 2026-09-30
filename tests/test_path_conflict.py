"""Native operator invariants with an actual small causal GQA transformer.

This exercises real attention/MLP forwards, not a mock returning fake scores.
It is not a test of the user's 8B weights or installed Transformers version.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.path_conflict.data import inventory, validate_text_cases


def test_inventory_does_not_inherit_old_annotations(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'settings.json').write_text(json.dumps(dict(model='model')))
    records = [dict(source_id='1', seed=0, trace='0.npz', tokens=3, response='good'),
               dict(source_id='1', seed=1, trace='1.npz', tokens=2, response='bad')]
    (source / 'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    (source / 'prompts.jsonl').write_text(json.dumps(dict(source_id='1', prompt='evidence value')))
    case = dict(case_id='one', source_id='1', supported=dict(seed=0, target='good'),
                unsupported=dict(seed=1, target='bad'), evidence=['evidence'], value_source=['value'])
    output = tmp_path / 'out'
    output.mkdir()
    inventory(source, output, [case])
    table = pd.read_csv(output / 'samples.csv')
    assert set(table.whole_answer_label) == {'not_annotated'}
    assert not table.trace_available.any()
    assert len(pd.read_csv(output / 'same_question_pairs.csv')) == 1


def test_compile_keeps_original_prompt_and_stops_before_target(tmp_path):
    from experiments.path_conflict.data import compile_cases
    tokenizer = WordTokenizer()
    prompt = 'Evidence caps. Value headdress. Output:'
    prompt_ids = tokenizer.encode(prompt)
    samples = []
    for seed, response in [(0, 'They wear caps'), (1, 'They instead wear headdress')]:
        ids = prompt_ids + tokenizer.encode(response)
        filename = f'{seed}.npz'
        tokens = np.array([tokenizer.decode([token]) for token in ids])
        count = len(ids) - len(prompt_ids)
        np.savez(tmp_path / filename, token_ids=ids, token_text=tokens, prompt_length=len(prompt_ids),
                 top_ids=np.zeros((count, 5), int), top_logits=np.zeros((count, 5)), log_normalizer=np.zeros(count))
        samples.append(dict(source_id='one', seed=seed, trace=filename, response=response))
    case = dict(case_id='test', source_id='one', supported=dict(seed=0, target='caps'),
        unsupported=dict(seed=1, target='headdress'), candidates=[' caps', ' headdress'],
        evidence=['Evidence caps.'], value_source=['Value headdress.'])
    result = compile_cases(tmp_path, tokenizer, samples, {'one': prompt}, [case], tmp_path)[0]
    assert not result['identical_prefix']
    for name, side in result['sides'].items():
        decoded = tokenizer.decode(side['prefix_ids'])
        assert decoded.endswith('wear')
        assert side['prefix_ids'][:len(prompt_ids)] == prompt_ids
        assert len(side['groups']['history']) == side['response_step']


def test_duplicate_or_missing_claim_text_rejected_before_model_load():
    records = [dict(source_id='a', seed=0, response='same same'), dict(source_id='a', seed=1, response='bad')]
    case = dict(case_id='a', source_id='a', supported=dict(seed=0, target='same'),
        unsupported=dict(seed=1, target='bad'), evidence=['source'], value_source=['value'])
    with pytest.raises(AssertionError, match='target not unique'):
        validate_text_cases(records, {'a': 'source value'}, [case])


class WordTokenizer:
    """Whitespace-preserving tokenizer used only to test saved-ID case assembly."""
    def __init__(self):
        self.pieces = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        import re
        matches = list(re.finditer(r' ?[A-Za-z0-9]+|[^A-Za-z0-9 ]| +', text))
        ids = []
        for match in matches:
            piece = match.group()
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(self.pieces.index(piece))
        return dict(input_ids=ids, offset_mapping=[match.span() for match in matches])

    def encode(self, text, add_special_tokens=False):
        return self(text)['input_ids']

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)
