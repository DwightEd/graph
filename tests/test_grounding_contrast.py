import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).with_name('grounding_contrast.py')
    if not path.exists():
        path = Path(__file__).parents[1] / 'next_iteration' / 'grounding_contrast.py'
    spec = importlib.util.spec_from_file_location('grounding_contrast', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_current_sentence_does_not_consume_future():
    m = module()
    before, current = m.focus_prefix('First fact. Second wrong')
    assert before == 'First fact.'
    assert current == ' Second wrong'


def test_decimal_does_not_end_sentence():
    m = module()
    assert m.focus_prefix('It costs 3.50 dollars')[1] == 'It costs 3.50 dollars'


def test_current_punctuation_belongs_to_completed_sentence():
    m = module()
    assert m.focus_prefix('One. Two.')[1] == ' Two.'


def test_prompt_roles_are_not_factual_labels():
    m = module()
    args = ('SOURCE FACT', 'INSTRUCTION', 'Earlier claim.', 'Current claim')
    source = m.verification_messages(*args, arm='source')
    history = m.verification_messages(*args, arm='history')
    assert 'SOURCE FACT' in source[1]['content']
    assert 'SOURCE FACT' not in history[1]['content']
    assert 'Earlier claim.' in source[1]['content']
    assert 'not evidence' in source[0]['content']
    assert 'B' in source[0]['content'] and 'A' in source[0]['content']


def test_score_is_raw_log_odds():
    m = module()
    logits = np.array([[3., 1.], [-2., 2.]])
    np.testing.assert_array_equal(m.log_odds(logits), [-2., 4.])


def test_retrospective_mapping_declares_lookahead():
    m = module()
    risks = np.array([0., 1., 2., 3.])
    mapped, lookahead = m.retrospective_scores(['One', 'One.', 'One. Two', 'One. Two.'], risks)
    np.testing.assert_array_equal(mapped, [1., 1., 3., 3.])
    np.testing.assert_array_equal(lookahead, [1, 0, 1, 0])


def test_duplicate_inputs_rejected():
    m = module()
    with pytest.raises(ValueError, match='duplicate'):
        m.validate_rows([{'id': 'x'}, {'id': 'x'}])


def test_predictions_cannot_read_ground_truth():
    m = module()
    with pytest.raises(ValueError, match='label'):
        m.validate_rows([{'id': 'x', 'labels': []}])
