import json
import hashlib
import numpy as np
import pytest
from structural_detector.features import features,select_sources,ARMS
from structural_detector.experiment import annotations,targets,fit,predict,metrics


def test_future_extension_cannot_change_existing_features():
    rng=np.random.default_rng(5);x=rng.random((40,5))
    for cut in range(1,40):np.testing.assert_array_equal(features(x[:cut]),features(x)[:cut])


def test_no_cold_start_and_onset_is_not_an_input():
    x=np.array([[2.,-1.,.4,.0,.2],[1.,-2.,.2,.1,.3]])
    z=features(x);assert z.shape==(2,11)
    assert z[0,0]==2 and z[0,5]==-.4 and z[0,8]==0
    assert z[1,8]==pytest.approx(2*np.exp(-.25))
    assert z[1,10]==pytest.approx(z[1,8]*(-.1))


def test_recent_memory_expires_after_eight_steps():
    x=np.zeros((11,5));x[0,0]=4
    z=features(x);assert z[8,8]>0 and z[9,8]==0


def test_source_split_groups_generators_and_rejects_official_overlap():
    rows=[dict(id=str(i),source_id=str(i//2),official_split='train') for i in range(20)]
    a=select_sources(rows);b=select_sources(rows[::-1]);assert a==b
    assert all(a[str(i)]==a[str(i+1)] for i in range(0,20,2))
    with pytest.raises(ValueError):select_sources(rows+[dict(id='t',source_id='0',official_split='test')])


def test_annotation_loader_does_not_parse_other_labels(tmp_path):
    p=tmp_path/'a.jsonl';p.write_text('{"id":"1", "labels":[]}\n{"id":"2", INVALID_UNRELATED_LABELS}\n')
    assert annotations(p,{'1'})=={'1':{'id':'1','labels':[]}}
    with pytest.raises(ValueError):annotations(p,{'3'})


def test_first_token_and_multiple_error_spans_are_retained():
    text='bad ok bad';row=dict(source_id='s',official_split='test',response_sha256=hashlib.sha256(text.encode()).hexdigest())
    gold=dict(source_id='s',split='test',response=text,labels=[dict(start=0,end=3),dict(start=7,end=10)])
    y=targets(row,np.array([[0,3],[3,7],[7,10]]),gold)
    assert y['error'].tolist()==[1,0,1] and y['onset'].tolist()==[1,0,1]


def test_calibrated_fit_probability_and_direction():
    rng=np.random.default_rng(9);x=rng.normal(size=(100,2));y=(x[:,0]>0).astype(int)
    m=fit(x[:70],y[:70],np.ones(70),x[70:],y[70:],np.ones(30))
    p=predict(m,np.array([[-2.,0.],[2.,0.]]));assert 0<=p[0]<p[1]<=1
    assert metrics(np.array([0,1]),p,True)['auroc']==1


def test_ablation_families_do_not_reintroduce_removed_cues():
    assert set(ARMS['without_entropy'])=={2,3,4,5,7}
    assert set(ARMS['without_attention'])=={0,1,6,8}
