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
