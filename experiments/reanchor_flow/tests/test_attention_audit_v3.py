"""Counterexamples for the mechanisms the full-data pass must distinguish."""
import json
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.attention_audit import AuditConfig, AuditObserver, capture_audit, reconstruct_attention
from experiments.reanchor_flow.attention_audit_stats import (
    by_correction, carrier_entry, incoming_reads, joint_tables, matched_joint_gap,
    matched_positions, bracket_positions, matched_difference, read_metrics, target_chain_tables, Moments,
)


def collect(a, tmp_path, chunk=3, special=None, start=4):
    l,h,n,_ = a.shape
    ids = np.arange(n)
    special = np.arange(n)==0 if special is None else special
    evidence = np.arange(n)==2
    unit = np.where(evidence,0,-1)
    history = tmp_path/f'history-{chunk}.npz'
    with ZipFile(history,'w') as archive:
        observer = AuditObserver(l,h,ids,start,evidence,special,unit,AuditConfig(top_k=3),archive)
        for layer in range(l):
            for q in range(0,n,chunk):
                observer.observe_rows(layer,q,a[layer,:,q:q+chunk],torch.ones(h,n))
        result = observer.finish()
    result['head_margin'] = np.zeros((l,h,n-start+1),np.float32)
    result['token_text'] = np.array(['word']*n)
    result['sample_id'] = np.array('fixture')
    return result,history


def random_attention(l=3,h=2,n=17):
    torch.manual_seed(3)
    a = torch.randn(l,h,n,n)
    return a.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),-torch.inf).softmax(-1)


def test_bos_dominance_is_excluded_without_renormalizing_the_model(tmp_path):
    a = torch.eye(12)[None,None].repeat(2,2,1,1)
    a[:,:,3:] = 0
    a[:,:,3:,0] = .9
    a[:,:,3:,2] = .1
    trace,_ = collect(a,tmp_path)
    np.testing.assert_allclose(trace['mass'].sum(-1),1,atol=1e-7)
    np.testing.assert_allclose(trace['mass'][...,0],.9)
    np.testing.assert_allclose(trace['ordinary_mass'],.1)
    expected=np.broadcast_to(np.arange(3,12)-2,trace['distance'][...,3].shape)
    np.testing.assert_allclose(trace['distance'][...,3],expected,atol=1e-6)
    assert not (trace['top_position']==0).any()
    np.testing.assert_array_equal(trace['top_position'][...,1,0],2)
    np.testing.assert_allclose(read_metrics(trace)['evidence_share'],1,atol=1e-6)
    assert not carrier_entry(trace)[0].any()


def test_no_ordinary_mass_is_missing_not_normal_or_zero_distance(tmp_path):
    a = torch.zeros(2,2,12,12); a[:,:,:,0]=1
    trace,path = collect(a,tmp_path)
    assert np.isnan(trace['distance']).all()
    assert np.isnan(trace['entropy']).all()
    assert (trace['top_position']==-1).all()
    with np.load(path) as h:
        incoming = incoming_reads(h['L0'],trace['ordinary_mass'][0],trace,((1,3),))
    assert not incoming['count'].any()
    assert np.isnan(incoming['fai']).all()


@pytest.mark.parametrize('chunk',[1,5,17])
def test_all_head_metrics_and_history_are_chunk_invariant(tmp_path,chunk):
    a=random_attention()
    left,lp=collect(a,tmp_path,chunk=2)
    right,rp=collect(a,tmp_path,chunk=chunk)
    for key in ('mass','distance','message_distance','entropy','change_tv','top_attention','unit_mass'):
        np.testing.assert_allclose(left[key],right[key],atol=2e-6,equal_nan=True)
    with np.load(lp) as x,np.load(rp) as y:
        for layer in range(a.shape[0]):
            np.testing.assert_array_equal(x[f'L{layer}'],y[f'L{layer}'])


def test_future_exposure_and_target_shift_match_dense_reference(tmp_path):
    a=random_attention(n=14)
    special=np.arange(14)==0; special[10]=True
    trace,path=collect(a,tmp_path,special=special)
    with np.load(path) as archive:
        result=incoming_reads(archive['L1'],trace['ordinary_mass'][1],trace,((1,3),(1,0)))
    for b in range(4,14):
        queries=[q for q in range(b+1,min(b+4,13)) if not special[q] and not special[q+1]]
        if queries and not special[b]:
            expected=np.stack([a[1,:,q,b].numpy()/trace['ordinary_mass'][1,:,q-3] for q in queries]).mean(0)
            np.testing.assert_allclose(result['fai'][:,b-4,0],expected,atol=1e-7)
            np.testing.assert_equal(result['count'][:,b-4,0],len(queries))
        else:
            assert np.isnan(result['fai'][:,b-4,0]).all()


def test_four_states_include_failed_entry_and_failed_reuse():
    entry=np.array([[[0,1,0,1]],[[0,0,0,0]]],float)
    reuse=np.array([[[0,0,0,0]],[[0,0,1,1]]],float)
    table=joint_tables(entry,reuse,np.zeros(4),np.ones(4,bool),1)
    np.testing.assert_array_equal(table[0,:,0,1],.25)
    assert np.isnan(table[1]).all()
    assert np.isnan(table[:,:,1,0]).all()
    assert table[0,:,0,1].sum()==1
    delta=matched_joint_gap(entry,reuse,np.array([[1,0],[3,2]]),1)
    np.testing.assert_allclose(delta[:,0,1],[-.5,.5,-.5,.5])


def test_matching_is_same_class_nearby_no_replacement_and_special_excluded():
    labels=np.array([0,1,1,0,0,1,0])
    classes=np.array([2,2,1,1,2,2,2])
    valid=np.array([1,1,1,1,0,1,1],bool)
    pairs=matched_positions(labels,classes,valid,max_gap=2)
    assert pairs.tolist()==[[1,0],[2,3],[5,6]]
    assert len(set(pairs[:,1]))==len(pairs)
    assert not matched_positions(np.ones(4),np.zeros(4),np.ones(4,bool),4).size


def test_target_chain_uses_real_carrier_and_deeper_layer(tmp_path):
    a=torch.eye(10)[None,None].repeat(2,1,1,1)
    # writer at b=5 reads evidence s=2, reader at q=7 reads carrier b=5.
    a[0,0,5]=0; a[0,0,5,2]=1
    a[1,0,7]=0; a[1,0,7,5]=1
    trace,path=collect(a,tmp_path,start=4,special=np.arange(10)==0)
    labels=np.array([0,0,0,0,1,0]) # token 8, predicted at q=7
    pairs=np.array([[4,3]])
    with np.load(path) as history:
        result=target_chain_tables(trace,history,labels,pairs,np.ones(6,bool))
    assert result['matched'][0,1]==pytest.approx(1)
    assert np.isnan(result['matched'][1,0])
    # Destroy the origin connection while keeping the downstream read unchanged.
    a[0,0,5]=0; a[0,0,5,4]=1
    trace,path=collect(a,tmp_path,start=4,special=np.arange(10)==0)
    with np.load(path) as history:
        changed=target_chain_tables(trace,history,labels,pairs,np.ones(6,bool))
    assert changed['matched'][0,1]==0


def test_source_moments_and_dependent_multiple_comparison_control():
    x=Moments()
    for v in (1.,2.,3.): x.add(np.array([v,np.nan]))
    result=x.finish(inference=True)
    np.testing.assert_array_equal(result['sources'],[3,0])
    assert result['mean'][0]==2
    assert result['ci95'][0,0]<2<result['ci95'][1,0]
    p=np.array([.001,.1,np.nan,.8])
    q=by_correction(p)
    assert np.all(q[np.isfinite(p)]>=p[np.isfinite(p)])
    assert np.isnan(q[2])
    constant=Moments()
    for _ in range(3): constant.add(np.array([1.]))
    uncertainty=constant.finish(inference=True)
    assert np.isnan(uncertainty['p']).all() and np.isnan(uncertainty['ci95']).all()


def test_two_sided_controls_remove_linear_position_drift():
    labels=np.array([0,0,1,1,1,0,0])
    pairs=bracket_positions(labels,np.zeros(7),np.ones(7,bool),max_gap=5)
    assert len(pairs)==3
    values=np.arange(7,dtype=float)*.73+1.41
    np.testing.assert_allclose(matched_difference(values,pairs),0,atol=1e-14)
    values[labels==1]+=2
    np.testing.assert_allclose(matched_difference(values,pairs),2,atol=1e-14)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_native_tiny_llama_full_history_writes_and_ordinary_edges(tmp_path,dtype):
    from transformers import LlamaConfig,LlamaForCausalLM
    cfg=LlamaConfig(vocab_size=41,hidden_size=24,intermediate_size=48,num_hidden_layers=3,
                    num_attention_heads=3,num_key_value_heads=1)
    cfg._attn_implementation='eager'
    torch.manual_seed(23)
    model=LlamaForCausalLM(cfg).to(dtype).eval()
    ids=np.arange(19); start=5
    evidence=np.isin(ids,[2,3]); special=np.isin(ids,[0,18]); units=np.where(evidence,0,-1)
    path=tmp_path/'sample.npz'
    result=capture_audit(model,ids,start,evidence,special,units,path,AuditConfig(top_k=3,query_chunk=4))
    np.savez_compressed(path, **result)
    with torch.no_grad():
        native=model(torch.tensor(ids)[None],output_attentions=True,use_cache=False)
    with np.load(path.with_suffix('.history.npz')) as history:
        for l in range(3):
            np.testing.assert_allclose(history[f'L{l}'],native.attentions[l][0,:,start-1:,start-1:].float(),atol=1e-7)
            for head in range(3):
                np.testing.assert_allclose(reconstruct_attention(path,l,head),native.attentions[l][0,head,start-1:].float(),atol=1e-7)
    assert not (result['top_position']==0).any()
    assert result['head_margin'].shape==(3,3,15)
    tol=3e-7 if dtype==torch.float32 else .005
    np.testing.assert_allclose(result['head_margin'].sum(1),result['attention_margin'],atol=tol,equal_nan=True)
    np.testing.assert_allclose(result['residual_margin'][-1,:-1],result['observed_margin'][:-1],atol=tol)
    assert np.max(np.abs(result['rounding_margin'][...,:-1]))<tol
    with np.load(path.with_suffix('.states.npz')) as states:
        assert states['value_0'].shape==(1,19,8) # native KV heads, not four duplicate copies
        assert states['head_0'].shape==(3,15,8)


def test_all_tasks_full_pipeline_balanced_labels_resume_and_offline(tmp_path,monkeypatch):
    import transformers
    from experiments.reanchor_flow.attention_audit_run import parser,run
    from experiments.reanchor_flow.tests.test_scan_dataset import _capture
    from experiments.reanchor_flow.tests.test_attention_rhythm_pipeline import CharacterTokenizer
    import experiments.reanchor_flow.scan_dataset as scans_module

    scans,output=tmp_path/'scans',tmp_path/'out'
    scans.mkdir()
    prompt="{'x': 1}"
    p,t=len(prompt)+2,24
    info=[]
    for split in ('train','test'):
        records={f'{split}-{task}-{case}':(f'{split}-{task}-{case}',task)
                 for task in ('QA','Summary','Data2txt') for case in ('negative','positive')}
        root=_capture(scans,records=records,overrides={'response_start':p,'sequence_length':p+t,
                'full_response_tokens':t,'processed_response_tokens':t,
                'route_row_position':np.arange(p-1,p+t-1),'token_ids':np.arange(1,p+t+1)})
        if split=='train': root.rename(scans/split)
        root=scans/split
        m=json.loads((root/'run_manifest.json').read_text()); m['config']['split']=split
        (root/'run_manifest.json').write_text(json.dumps(m))
        for _,(source,task) in records.items():
            info.append(dict(source_id=source,task_type=task,prompt=prompt,
                             source_info={'passages':prompt} if task=='QA' else prompt))
    source_file=tmp_path/'sources.jsonl'; source_file.write_text('\n'.join(map(json.dumps,info)))
    tokenizer=CharacterTokenizer(); tokenizer.all_special_ids=[1,p+t]
    cfg=transformers.LlamaConfig(vocab_size=101,hidden_size=24,intermediate_size=48,
                                num_hidden_layers=3,num_attention_heads=3,num_key_value_heads=1)
    cfg._attn_implementation='eager'
    model=transformers.LlamaForCausalLM(cfg).eval()
    loads=[]
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',lambda *a,**kw:cfg)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',lambda *a,**kw:loads.append(1) or model)
    class Labels:
        def __init__(self,*a,**kw): pass
        def load(self,sample):
            y=np.zeros(t,int)
            if sample.sample_id.endswith('positive'): y[8:10]=1
            return y
    monkeypatch.setattr(scans_module,'ScanLabelStore',Labels)
    argv=['--scans',str(scans),'--source-info',str(source_file),'--output',str(output),
          '--device','cpu','--dtype','float32','--query-chunk','5','--top-k','3',
          '--horizon','1:3','--onset-radius','2','--plots-per-class','1']
    # Preflight must finish without loading the LLM.
    run(parser().parse_args([*argv,'--plan-only']))
    assert not loads
    report=run(parser().parse_args(argv))
    assert loads==[1]
    assert len(report['groups'])==8
    assert report['groups']['test/ALL']['hallucinated_tokens']==6
    assert report['groups']['test/ALL']['special_targets']==6
    assert report['groups']['test/ALL']['negative_answers']==3
    assert report['groups']['test/ALL']['positive_answers']==3
    assert len(list(output.rglob('*.review.html')))==12
    assert len(list(output.rglob('*.states.npz')))==12
    with np.load(output/'cohorts'/'test_QA.npz') as data:
        assert data['raw_mean'].shape[-2:]==(3,3)
        assert data['joint_raw_mean'].shape==(2,4,9,9)
        assert data['chain_matched_mean'].shape==(9,9)
    def forbidden(*a,**kw): raise AssertionError('must reuse completed capture')
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',forbidden)
    manifest=run(parser().parse_args([*argv,'--phase','capture']))
    assert all(e['resumed'] for e in manifest['samples'])
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',forbidden)
    monkeypatch.setattr(transformers.AutoConfig,'from_pretrained',forbidden)
    monkeypatch.setattr(scans_module,'ScanLabelStore',forbidden)
    # Disable repeated figures; exercise model-free changed-horizon analysis.
    import experiments.reanchor_flow.attention_audit_plot as plotting
    monkeypatch.setattr(plotting,'plot_cohort',lambda *a,**kw:None)
    again=run(parser().parse_args(['--phase','analyze','--output',str(output),
             '--horizon','1:2','--plots-per-class','0','--onset-radius','1']))
    assert again['groups']['train/ALL']['hallucinated_tokens']==6
    # An interrupted run keeps planned samples in index.json. Ignore temporary
    # files, reject missing companions, and do not shrink that resume index.
    index=output/'index.json'
    manifest=json.loads(index.read_text())
    manifest['samples'][0]['resumed']=False  # final NPZ exists despite stale flag
    for i,e in enumerate(manifest['samples'][-2:]):
        path=output/e['path']
        incomplete=path if i==0 else path.with_suffix('.qk.npz')
        incomplete.rename(incomplete.with_suffix('.tmp.npz'))
        path.with_suffix('.labels.npz').unlink()
        path.with_suffix('.input.npz').unlink()
    index.write_text(json.dumps(manifest))
    original=index.read_bytes()
    offline=['--phase','analyze','--output',str(output),'--horizon','1:2',
             '--plots-per-class','0','--onset-radius','1']
    with pytest.raises(ValueError,match='completed-only'):
        run(parser().parse_args(offline))
    partial=run(parser().parse_args([*offline,'--completed-only']))
    assert index.read_bytes()==original
    coverage=partial['analysis_coverage']
    assert (coverage['completed_samples'],coverage['planned_samples'],coverage['skipped_samples'])==(10,12,2)
    assert coverage['partial'] is True
    assert coverage['groups']['test/Data2txt']['completed_samples']==0
    assert 'test/Data2txt' not in partial['groups']
    assert 'Data2txt' not in partial['replication']
    assert partial['groups']['test/ALL']['samples']==4
    assert len(json.loads((output/'review_index.json').read_text()))==10
    assert '部分采集结果' in (output/'summary.md').read_text(encoding='utf-8')
    assert 'Partial capture' in (output/'gallery.html').read_text()


def test_empty_interrupted_capture_does_not_load_model_or_rewrite_index(tmp_path,monkeypatch):
    import transformers
    from experiments.reanchor_flow.attention_audit import SCHEMA
    from experiments.reanchor_flow.attention_audit_run import parser,run
    def forbidden(*a,**kw): raise AssertionError('analysis must not load model/tokenizer')
    monkeypatch.setattr(transformers.AutoModelForCausalLM,'from_pretrained',forbidden)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',forbidden)
    manifest=dict(audit_schema=SCHEMA,labels_used_for_capture=False,settings={},config={},
                  samples=[dict(path='train/QA/a.npz',split='train',task_type='QA',response_tokens=10)])
    index=tmp_path/'index.json'; index.write_text(json.dumps(manifest))
    before=index.read_bytes()
    with pytest.raises(ValueError,match='no completed samples'):
        run(parser().parse_args(['--phase','analyze','--completed-only','--output',str(tmp_path)]))
    assert index.read_bytes()==before
    with pytest.raises(ValueError,match='only valid'):
        run(parser().parse_args(['--phase','all','--completed-only','--output',str(tmp_path)]))
