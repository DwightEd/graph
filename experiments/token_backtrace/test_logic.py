import numpy as np

from .logic import apply_constraints, interval, interval_relation, polarity_claims, duration_claims


def test_open_bounds_and_one_way_entailment():
    assert interval_relation(interval('more than', 4), interval('', 4)) == 'contradiction'
    assert interval_relation(interval('at least', 4), interval('', 4)) == 'unknown'
    assert interval_relation(interval('', 4), interval('at least', 4)) == 'supported'
    assert interval_relation(interval('at most', 4), interval('at least', 4)) == 'unknown'


def test_polarity_scope_conflict_and_role():
    assert polarity_claims('The store is not open.', 'The store is open.')[0]['status'] == 'contradiction'
    assert polarity_claims('The store is not open.', 'The store is not open.')[0]['status'] == 'supported'
    assert polarity_claims('The school is not open.', 'The store is open.') == []
    assert polarity_claims('The store is not open. The store is open.', 'The store is open.') == []
    assert polarity_claims('If it rains the store is not open.', 'The store is open.') == []
    assert polarity_claims('The store is not open on Sundays.', 'The store is open.') == []


def test_duration_controls_and_alignment_abstention():
    source = 'For more than three weeks, the sailor was stranded on a remote island.'
    wrong = 'The sailor was stranded on the remote island for three weeks.'
    correct = 'The sailor was stranded on the remote island for more than three weeks.'
    assert duration_claims(source, wrong)[0]['status'] == 'contradiction'
    assert duration_claims(source, correct)[0]['status'] == 'supported'
    assert duration_claims(source, 'The opera played for three weeks.') == []
    assert duration_claims(source + ' Another boat travelled for two weeks.', wrong) == []


def test_unknown_preserves_native_and_no_averaging():
    native = np.array([.1, .4, .8])
    np.testing.assert_array_equal(apply_constraints(native, np.array([1, 0, -1])), [1., .4, 0.])


def relation(**changes):
    item = dict(subject='Alice', predicate='helps', object='Bob', polarity='positive',
                condition='', modality='asserted', speech_act='assert', quantity=None)
    item.update(changes)
    return item


def test_grounded_parser_rejects_fabrication_and_preserves_ambiguity():
    import json
    from .logic import grounded_relations, relation_consensus

    text = 'Alice helps Bob. Alice helps Bob.'
    record = dict(quote='Alice helps Bob.', occurrence=1, alternatives=[relation()])
    records, rejected = grounded_relations(text, json.dumps([record]))
    assert not rejected and records[0]['start'] == 17
    fabricated = dict(record, quote='Alice dislikes Bob.')
    assert grounded_relations(text, json.dumps([fabricated]))[1][0]['reason'] == 'ungrounded_quote'
    assert grounded_relations(text, 'broken JSON')[1][0]['reason'] == 'invalid_json'
    ambiguous = dict(record, alternatives=[relation(), relation(subject='Carol')])
    assert relation_consensus([ambiguous], record)[0] == 'unknown'


def test_roles_quantities_conditions_quotes_are_not_collapsed_to_lexical_overlap():
    from .logic import compare_relations

    # Bob can also help Alice: role reversal alone is not a contradiction.
    assert compare_relations(relation(), relation(subject='Bob', object='Alice')) == 'unknown'
    taller = relation(predicate='taller than')
    assert compare_relations(taller, relation(subject='Bob', object='Alice', predicate='taller than')) == 'contradiction'
    assert compare_relations(relation(condition='if it rains'), relation()) == 'unknown'
    assert compare_relations(relation(), relation(speech_act='quote')) == 'unknown'
    quantity = dict(lower=4, lower_closed=False, upper=None, upper_closed=False, unit='day')
    exact = dict(lower=4, lower_closed=True, upper=4, upper_closed=True, unit='day')
    assert compare_relations(relation(quantity=quantity), relation(quantity=exact)) == 'contradiction'
    assert compare_relations(relation(quantity=quantity), relation(quantity=dict(exact, unit='week'))) == 'unknown'


def test_all_original_tokens_and_conflicting_source_evidence_remain_unknown():
    from .logic import parsed_token_constraints

    source = [dict(quote='Alice helps Bob.', alternatives=[relation()]),
              dict(quote='Alice does not help Bob.', alternatives=[relation(polarity='negative')])]
    answer = [dict(start=0, end=10, quote='Alice helps Bob.', alternatives=[relation()])]
    statuses, events = parsed_token_constraints(source, answer, [(0, 5), (5, 10), (11, 12)])
    assert statuses == ['unknown'] * 3
    assert events[0]['status'] == 'unknown'
    assert len(events[0]['bindings']) == 2


def test_possible_synonym_or_coreference_cannot_be_excluded_by_parser_key():
    from .logic import relation_consensus

    answer = dict(quote='Alice helps Bob.', alternatives=[relation()])
    source = [dict(quote='Alice helps Bob.', alternatives=[relation()]),
              dict(quote='She assists Bob.', alternatives=[relation(predicate='assists')])]
    status, bindings = relation_consensus(source, answer)
    assert status == 'unknown'
    assert all(row['status'] != 'excluded' for row in bindings)


def test_grounding_pointer_keeps_original_context_and_repeated_occurrence():
    from .logic import grounding_request

    answer = 'X is open. X is open.'
    request = grounding_request('X is closed.', answer, [(0, 1), (11, 12)], 1)
    assert 'X is open. <TARGET>X</TARGET> is open.' in request
    assert 'TARGET character range: [11, 12)' in request
    assert 'X is closed.' in request
    assert 'gold' not in request


def test_audit_pointer_keeps_response_intact_when_target_splits_a_word():
    from .logic import grounding_request

    answer = 'The vehicle is stopped.'
    request = grounding_request('A vehicle stopped.', answer, [(5, 8)], 0, 'audit', 'Fallible observation.')
    assert 'RESPONSE:\nThe vehicle is stopped.' in request
    assert '"substring": "ehi"' in request
    assert '<TARGET>' not in request
    assert 'Fallible observation.' in request


def test_grounding_conditional_score_does_not_hide_other_vocabulary_mass():
    import torch
    from .logic import grounding_probabilities

    logits = torch.tensor([[2., 1., 20.], [1., 2., 20.]])
    score, logp, mass = grounding_probabilities(logits, [0, 1])
    assert score[0] > .5 and score[1] < .5
    assert torch.all(mass < 1e-6)
    torch.testing.assert_close(logp.exp().sum(-1), mass)


def test_grounding_prefill_left_padding_matches_independent_requests():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from .logic_benchmark import grounding_batch

    vocabulary = {'<unk>': 0, '<pad>': 1, 'A': 2, 'B': 3, 'one': 4, 'two': 5, 'three': 6, '<bos>': 7}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='<bos> $A', special_tokens=[('<bos>', 7)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
        unk_token='<unk>', bos_token='<bos>', padding_side='left',
        chat_template="{{ bos_token }}{{ messages[1]['content'] }}")
    torch.manual_seed(17)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        _attn_implementation='sdpa')).eval()
    requests = ['one two', 'three one two three']
    batch = grounding_batch(model, tokenizer, requests, 'system')
    singles = [grounding_batch(model, tokenizer, [request], 'system') for request in requests]
    np.testing.assert_allclose(batch['choice_logp'], np.concatenate([r['choice_logp'] for r in singles]), atol=1e-6)
    assert batch['input_tokens'].tolist() == [3, 5]


def test_grounding_evaluation_checks_full_coverage_and_special_token_denominator(tmp_path, monkeypatch):
    import json
    import pytest
    from experiments.decision_risk_flow import data
    from experiments.automatic_evidence import evaluate
    from .logic_benchmark import grounding_evaluate

    response = dict(answer_ids=[4, 5, 128001], offsets=[[0, 1], [1, 2], [2, 2]],
                    token_text=['a', 'b', ''], special_ids=[128001])
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(records=[dict(key='x', original={}, response=response)])))
    (tmp_path / 'scores_frozen.json').write_text(json.dumps(dict(complete_answers=True)))
    (tmp_path / 'x').mkdir()
    monkeypatch.setattr(data, 'labels', lambda records: {'x': np.array([1, 0, 0])})

    def spans(rows, records, methods):
        assert [row['valid'] for row in rows] == [True, True, False]
        return {'checked': True}

    monkeypatch.setattr(evaluate, 'annotated_span_metrics', spans)
    np.savez(tmp_path / 'x/scores.npz', target=[0, 1, 2], token_ids=response['answer_ids'],
             score=[.9, .1, .99], choice_mass=[.8, .8, .8])
    grounding_evaluate(tmp_path)
    result = json.loads((tmp_path / 'evaluation.json').read_text())
    assert result['pooled']['tokens'] == 2 and result['pooled']['tp'] == 1
    assert result['pooled']['fp'] == 0 and result['span_metrics']['checked']
    np.savez(tmp_path / 'x/scores.npz', target=[0, 1], token_ids=[4, 5], score=[.9, .1], choice_mass=[.8, .8])
    with pytest.raises(AssertionError):
        grounding_evaluate(tmp_path)


def test_grounding_reuse_copies_only_complete_original_answers(tmp_path):
    import json
    import pytest
    from .logic_benchmark import reuse_grounding_answer

    previous, output = tmp_path/'previous', tmp_path/'output'
    (previous/'x').mkdir(parents=True)
    output.mkdir()
    row = dict(key='x', response=dict(answer_ids=[4, 5]))
    assert not reuse_grounding_answer(previous, row, output)
    (previous/'x/complete.json').write_text(json.dumps(dict(tokens=2, forward_calls=100, seconds=4.)))
    np.savez(previous/'x/scores.npz', target=[0, 1], token_ids=[4, 5], score=[.9, .1])
    with pytest.raises(AssertionError):
        reuse_grounding_answer(previous, row, output, expect_memo=True)
    (previous/'x/memo.json').write_text(json.dumps(dict(complete=False)))
    with pytest.raises(AssertionError):
        reuse_grounding_answer(previous, row, output, expect_memo=True)
    (previous/'x/memo.json').write_text(json.dumps(dict(complete=True)))
    original = (previous/'x/scores.npz').read_bytes()
    assert reuse_grounding_answer(previous, row, output, expect_memo=True)
    assert (output/'x/scores.npz').read_bytes() == original
    saved = json.loads((output/'x/complete.json').read_text())
    assert saved['forward_calls'] == 100 and saved['new_forward_calls'] == 0 and saved['new_seconds'] == 0
    assert 'reused_from' not in json.loads((previous/'x/complete.json').read_text())
    bad_row = dict(key='x', response=dict(answer_ids=[4, 6]))
    with pytest.raises(AssertionError):
        reuse_grounding_answer(previous, bad_row, output)


def test_grounding_reuse_requires_identical_manifest_and_protocol(tmp_path):
    import json
    import pytest
    from .logic_benchmark import grounding_reuse_records

    protocol = dict(prompt='fixed', memo_prompt='memo', batch_size=1, assistant_prefix='prefix',
                    grounding_version='audit', threshold=.5, max_new_tokens=1024)
    records = [dict(key='x', source='original')]
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(model='fixed-model', records=records)))
    assert grounding_reuse_records(tmp_path, records, protocol, 'fixed-model', 2048) == {'x': records[0]}
    for changed in ([], [dict(key='x', source='changed')], records + [dict(key='y')]):
        with pytest.raises(AssertionError):
            grounding_reuse_records(tmp_path, changed, protocol, 'fixed-model', 2048)
    with pytest.raises(AssertionError):
        grounding_reuse_records(tmp_path, records, dict(protocol, prompt='different'), 'fixed-model', 2048)


def test_structured_boolean_binding_negation_and_coordination():
    from .logic import structured_claims

    source = repr(dict(name='West Library', attributes={'WiFi': 'no', 'OutdoorSeating': True,
        'Ambience': {'quiet': True, 'romantic': False}}, hours={}))
    claims = structured_claims(source, 'Outdoor seating and WiFi are available, with a quiet and romantic atmosphere.')
    assert [(c['witnesses'][0]['path'], c['status']) for c in claims] == [
        ('attributes.WiFi', 'contradiction'), ('attributes.OutdoorSeating', 'supported'),
        ('attributes.Ambience.romantic', 'contradiction')]
    assert structured_claims(source, 'This is casual dining.') == []
    assert structured_claims(source, 'A quiet atmosphere.')[0]['status'] == 'supported'
    assert structured_claims(source, 'No WiFi is available.')[0]['status'] == 'supported'
    assert structured_claims(source, 'WiFi is not available.')[0]['status'] == 'supported'
    assert structured_claims(source, 'The library does not offer WiFi.') == []
    assert structured_claims(source, 'If you visit, WiFi may be available.') == []
    assert structured_claims(source, 'The claim "WiFi is available" is false.') == []
    conflict = repr(dict(attributes={'WiFi': 'no'}, review_info=[{'review_text': 'WiFi was great.'}]))
    assert structured_claims(conflict, 'WiFi is available.') == []


def test_structured_hours_scope_minutes_and_corrected_controls():
    from .logic import structured_claims, DAYS, clock_minutes

    hours = {day: '6:15-13:45' for day in DAYS}
    source = repr(dict(hours=hours))
    assert clock_minutes('12:00 AM') == 0
    assert clock_minutes('12:00 PM') == 720
    assert structured_claims(source, 'Open Monday to Sunday from 6:15 AM to 1:45 PM.')[0]['status'] == 'supported'
    assert structured_claims(source, 'Open seven days a week from 6:15 AM to 1:45 AM.')[0]['status'] == 'contradiction'
    assert structured_claims(source, 'Open from 6:15 AM to 1:45 PM.') == []
    hours['Sunday'] = '8:15-13:45'
    mixed = repr(dict(hours=hours))
    assert structured_claims(mixed, 'Open Monday to Sunday from 6:15 AM to 1:45 PM.')[0]['status'] == 'contradiction'
    assert structured_claims(mixed, 'Open Monday to Saturday from 6:15 AM to 1:45 PM.')[0]['status'] == 'supported'


def test_passage_denial_requires_same_number_action_and_object():
    from .logic import passage_denial_claims

    source = 'passage 1:Step 1: Fold the shirt. passage 2:Step 2: Fold the blanket.'
    answer = 'Passage 2 does not provide instructions for folding a blanket.'
    claim = passage_denial_claims(source, answer)[0]
    assert claim['witnesses'][0]['quote'] in source
    assert passage_denial_claims(source, answer.replace('2', '1')) == []
    assert passage_denial_claims(source, answer.replace('folding', 'washing')) == []
    assert passage_denial_claims(source, 'The false claim "' + answer + '" is rejected.') == []


def test_witness_separates_disclaimer_heuristic_and_unknown():
    from .logic import witness_token_constraints

    answer = '1. Bake for 17 minutes (unspecified in passages)\nAn ordinary introduction.'
    offsets = [(0, 1), (12, 14), (50, 58)]
    status, strict, heuristic, claims = witness_token_constraints('The pan is clean.', answer, offsets)
    assert status.tolist() == [1, 1, 0]
    assert strict.tolist() == [False, False, False]
    assert heuristic.tolist() == [True, True, False]
    assert claims[0]['witnesses'] == [] and claims[0]['status'] == 'unsupported_heuristic'
    assert witness_token_constraints('The pan is clean.', 'An ordinary introduction.', [(0, 2)])[0].tolist() == [0]


def test_structured_missing_null_and_invalid_evidence_abstains():
    from .logic import structured_claims, DAYS

    answer = 'WiFi and a romantic atmosphere. Open Monday to Sunday from 6:15 AM to 1:45 PM.'
    for source in ['{broken', repr({'attributes': None, 'hours': None}),
                   repr({'attributes': {'Ambience': None}, 'review_info': None}),
                   repr({'hours': {day: '25:00-13:45' for day in DAYS}}),
                   repr({'hours': {day: None for day in DAYS}})]:
        assert structured_claims(source, answer) == []
    source = repr({'attributes': {'WiFi': 'no'}, 'review_info': [None, {}, {'review_text': None}]})
    assert structured_claims(source, 'WiFi is available.')[0]['status'] == 'contradiction'


def test_witness_capture_rejects_partial_misaligned_cache(tmp_path, monkeypatch):
    import json
    import pytest
    from types import SimpleNamespace
    from . import logic_benchmark as benchmark

    previous = tmp_path/'previous'
    (previous/'case').mkdir(parents=True)
    row = dict(key='case', source=dict(prompt_with_source=[1], source_mask=[True]),
        response=dict(text='A B', answer_ids=[10, 11], offsets=[[0, 1], [2, 3]]))
    (previous/'manifest.json').write_text(json.dumps(dict(model='stub', records=[row])))
    (previous/'scores_frozen.json').write_text(json.dumps(dict(status='complete', complete_answers=True, answers=1)))
    (previous/'protocol.json').write_text(json.dumps(dict(grounding_version='pointer', target_limit=None)))
    monkeypatch.setattr(benchmark.AutoTokenizer, 'from_pretrained',
        lambda *args, **kwargs: SimpleNamespace(decode=lambda ids: 'The store is closed.'))
    for name, targets, ids, score in [('missing', [0], [10], [.1]),
        ('wrong_id', [0, 1], [10, 12], [.1, .2]), ('nonfinite', [0, 1], [10, 11], [.1, np.nan])]:
        np.savez(previous/'case'/'scores.npz', target=targets, token_ids=ids, score=score)
        with pytest.raises(AssertionError):
            benchmark.witness_capture(previous, tmp_path/name)
        assert not (tmp_path/name/'scores_frozen.json').exists()
