import numpy as np

from next_iteration.internal_error_units import fixtures, token_targets
from next_iteration.internal_error_units_capture import groups_for_queries, reverse_history_indices


def test_factorial_identical_draft_opposite_binding_balanced():
    rows, gold = fixtures()
    assert len(rows) == 768
    groups = {}
    for r in rows:
        groups.setdefault((r['source_id'], gold[r['id']]['pair_id']), []).append(r)
    assert len(groups) == 384
    for pair in groups.values():
        assert len(pair) == 2
        assert pair[0]['response'] == pair[1]['response']
        assert pair[0]['source'] != pair[1]['source']
        assert {gold[r['id']]['condition'] for r in pair} == {'supported', 'misbound'}
        assert sorted(pair[0]['source'].split()) == sorted(pair[1]['source'].split())


def test_source_groups_never_cross_splits():
    rows, _ = fixtures()
    split = {}
    for r in rows:
        split.setdefault(r['source_id'], set()).add(r['split'])
    assert len(split) == 96 and all(len(v) == 1 for v in split.values())
    assert {s:sum(r['split']==s for r in rows) for s in ['train','calibration','test']} == {
        'train':384, 'calibration':128, 'test':256}


def test_every_source_has_complete_balanced_eight_cell_factorial():
    rows,gold=fixtures()
    for sid in {r['source_id'] for r in rows}:
        rr=[r for r in rows if r['source_id']==sid]
        assert len(rr)==8
        assert sum(gold[r['id']]['condition']=='misbound' for r in rr)==4
        assert len({r['source'] for r in rr})==4
        assert len({r['response'] for r in rr})==2
        for value in {gold[r['id']]['target_value'] for r in rr}:
            conditions=[gold[r['id']]['condition'] for r in rr if gold[r['id']]['target_value']==value]
            assert conditions.count('misbound')==conditions.count('supported')==2


def test_core_onset_completion_are_not_the_same_target():
    offsets = [[0,3],[3,6],[6,9],[9,12],[12,15]]
    g = dict(core=[3,9], scope=[3,12], onset_token=2)
    y = token_targets(offsets, g)
    assert y['onset'].tolist() == [0,0,1,0,0]
    assert y['core'].tolist() == [0,0,1,0,0]
    assert y['scope'].tolist() == [0,0,1,1,0]


def test_first_token_error_is_retained_and_normal_is_not_positive():
    offsets = [[0,1],[1,4],[4,8]]
    y = token_targets(offsets, dict(core=[0,4],scope=[0,8],onset_token=0))
    assert y['onset'].tolist() == [1,0,0]
    assert y['core'].tolist() == [1,1,0]
    assert all(not v.any() for v in token_targets(offsets, {}).values())


def test_multiple_tokens_and_boundary_overlap():
    y = token_targets([[0,4],[4,8],[8,12]], dict(core=[3,9],scope=[3,12]))
    assert y['core'].tolist() == [1,1,1]
    assert y['onset'].sum() == 1


def test_masks_and_shuffles_never_use_future_or_self_as_history():
    p, total = 4, 10
    source, history = groups_for_queries(p,total,[False,True,True,False])
    reverse = reverse_history_indices(p,total)
    for t,q in enumerate(range(p,total)):
        old = np.flatnonzero(history[t])
        assert old.tolist() == list(range(p,q))
        assert sorted(reverse[t,old].tolist()) == old.tolist()
        assert reverse[t,old].tolist() == old[::-1].tolist()
        assert all(k < q for k in reverse[t,old])
    assert np.flatnonzero(source).tolist() == [1,2]


def test_prefix_extension_cannot_change_existing_masks():
    _, a = groups_for_queries(3,8,[False,True,False])
    _, b = groups_for_queries(3,12,[False,True,False])
    assert np.array_equal(a,b[:5,:8])
    assert np.array_equal(reverse_history_indices(3,8),reverse_history_indices(3,12)[:5,:8])


def test_all_construction_core_strings_and_scope_ends_valid():
    rows, gold = fixtures()
    for r in rows:
        g = gold[r['id']]
        a,b = g['candidate_value']
        assert r['response'][a:b] == g['target_value']
        u,v = g['unit']
        assert r['response'][v-1] == '.'
        assert u <= a < b < v < len(r['response'])
        assert (g['core'] is None) == (g['condition'] == 'supported')


def test_revisit_window_is_forward_only_and_cascade_counts_misses():
    from next_iteration.internal_error_units_evaluate import causal_window, summarize
    event=np.array([0,0,1,0,0,0],bool)
    assert causal_window(event,0).tolist()==[False,False,True,False,False,False]
    assert causal_window(event,2).tolist()==[False,False,True,True,True,False]
    rows=[dict(id='x')]
    labs={'x':dict(core=np.array([1,0,0,1,0,0]),onset=np.array([1,0,0,0,0,0]),scope=np.array([1,0,0,1,1,0]))}
    result=summarize(rows,{'x':np.full(6,.9)},labs,{'x':event},.5)
    assert result['windows']['0']['unconditional_recall_at_calibration_threshold']==0
    assert result['windows']['4']['unconditional_recall_at_calibration_threshold']==.5
    assert result['windows']['4']['onset_coverage']==0


def test_calibrated_readout_is_monotone_and_has_no_test_fit_argument():
    from next_iteration.internal_error_units_evaluate import fit_head, predict
    rng=np.random.default_rng(47)
    x=rng.normal(size=(160,2));y=(x[:,0]+.1*x[:,1]>0).astype(int)
    model=fit_head(x[:100],y[:100],np.ones(100),x[100:],y[100:],np.ones(60))
    p=predict(model,np.array([[-3.,0.],[0.,0.],[3.,0.]]))
    assert np.all(np.diff(p)>0)
    assert .05<=model['calibration'][0]<=20


def test_pair_cascade_keeps_same_draft_missed_member_and_source_cluster():
    from next_iteration.internal_error_units_evaluate import paired_binding
    rows=[dict(id='a',source_id='s',offsets=[[0,1],[1,2]]),dict(id='b',source_id='s',offsets=[[0,1],[1,2]])]
    gold={'a':dict(candidate_value=[1,2],condition='supported',pair_id='p'),
          'b':dict(candidate_value=[1,2],condition='misbound',pair_id='p')}
    probs={'a':np.array([0.,.2]),'b':np.array([0.,.8])}
    events={'a':np.array([0,1]),'b':np.array([0,0])}
    all_result,_,_=paired_binding(rows,probs,gold,events)
    result,_,_=paired_binding(rows,probs,gold,events,0)
    assert all_result['ordering_accuracy']['point']==1
    assert result['ordering_accuracy']['point']==0
    assert result['one_hit_pairs']==1 and result['both_hit_pairs']==0
    assert result['candidate_any_hit']['misbound']['point']==0
    assert result['sources']==1
