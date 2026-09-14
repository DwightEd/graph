import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from reuse_detector.core import (Config, source_splits, fit_reference, seed_scores, propagate,
                                 controlled_edges, validate_edges, score_trace)
from reuse_detector.capture import attention_blocks, capture_response
from reuse_detector.evaluation import Ranking, evaluate
from reuse_detector.run import (score, save_npz, write_json, digest, parser, run, load_roster)


def row(i='r', split='train', n=24):
    text = 'abcdefghijklmnopqrstuvwx'[:n]
    return dict(id=i, source_id=i, official_split=split, task='QA', generator='generator',
                offsets=[[t,t+1] for t in range(n)], response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_ids=[1,2,3,4]+list(range(5,n+5)), prompt_length=4, source_mask=[False,True,True,False])


def edges(n, c=2, w=4):
    return np.zeros((n,1,c,w))


def test_unlabeled_reference_ties_and_monotonicity():
    r = row(); cfg=Config(min_reference=1, per_source=512)
    ref=fit_reference([r],lambda _:np.ones(24),cfg)
    s=seed_scores(np.ones(12),r,ref)
    assert not s['seed'].any() and not s['onset_rank'].any()
    ref=fit_reference([r],lambda _:np.linspace(0,10,200),cfg)
    values=seed_scores(np.array([1.,4.,11.]),r,ref)
    assert np.isfinite(values['tail']).all() and (values['seed']>=0).all()
    # Compare inside a common positional reference bin.
    values=seed_scores(np.r_[np.zeros(64), [1.,4.,11.]],r,ref)
    assert np.all(np.diff(values['onset_rank'][-3:])>0)


def test_source_split_no_test_or_calibration_in_reference():
    rows=[row(str(i)) for i in range(10)]+[row('test','test')]
    split=source_splits(rows)
    assert split['test']=='test' and set(split.values())=={'reference','calibration','test'}
    assert source_splits(rows[::-1])==split
    with pytest.raises(ValueError): source_splits(rows+[row('test','train')])


def test_reference_source_balanced_not_long_answer_dominated():
    r1,r2=row('1'),row('2'); cfg=Config(min_reference=10000, per_source=500)
    ref=fit_reference([r1,r2],lambda r: np.zeros(10) if r['id']=='1' else np.full(100,10.),cfg)
    table=ref['tables']['QA|generator|-1']
    assert table['cumulative'][9]==pytest.approx(.5)
    s=seed_scores([5.],r1,ref)
    assert s['tail'][0]==pytest.approx((1+110*.5)/111)


def test_reference_cannot_fall_back_to_other_task():
    ref=fit_reference([row()],lambda _:np.ones(24),Config())
    r=row();r['task']='Summary'
    with pytest.raises(ValueError): seed_scores([1.,2.],r,ref)


def test_causal_prefix_invariance():
    rng=np.random.default_rng(3); n=40; cfg=Config(window=4)
    a=edges(n); a[:]=rng.random(a.shape)*.1
    for t in range(4):a[t,:,:,t:]=0
    seeds=rng.random(n)
    full=propagate(seeds,a,cfg,detail=True)
    for cut in (1,3,9,24):
        part=propagate(seeds[:cut],a[:cut],cfg,detail=True)
        for name in full:np.testing.assert_array_equal(full[name][:cut],part[name])


def test_no_seeds_no_spreading_and_no_edges_no_spreading():
    a=edges(30);a[1:,:,:,0]=1.;cfg=Config(window=4)
    assert not propagate(np.zeros(30),a,cfg)['risk'].any()
    s=np.zeros(30);s[2]=.7
    np.testing.assert_array_equal(propagate(s,edges(30),cfg)['risk'],s)


def test_multihop_local_relay_survives_beyond_window():
    n=40; a=edges(n);a[1:,:,:,0]=1.;s=np.zeros(n);s[0]=1;cfg=Config(window=4)
    r=propagate(s,a,cfg,detail=True)
    np.testing.assert_allclose(r['risk'],.9**np.arange(n))
    assert r['dominant_origin'][-1].tolist()==[0,0]
    assert propagate(s,a,cfg,recursive=False)['risk'][2]==0


def test_mass_sink_and_normal_endpoint_stop_propagation():
    a=edges(6); s=np.zeros(6);s[0]=1
    a[1,:,:,0]=.5 # remainder is prompt or unrelated sources
    a[2,:,:,0]=0 # reread prompt only, no support for carrying old risk
    a[3,:,:,0]=1 # read the now-unmarked y2, not the seed
    r=propagate(s,a,Config(window=4))
    assert r['risk'][1]==pytest.approx(.45) and r['risk'][2]==0 and r['risk'][3]==0


def test_same_mass_different_endpoints_produce_different_risk():
    s=np.array([1.,0.,0.]);a=edges(3);b=a.copy()
    a[2,:,:,1]=.8 # seed y0
    b[2,:,:,0]=.8 # unmarked y1
    x,y=(propagate(s,v,Config(window=4)) for v in (a,b))
    assert x['risk'][2]==pytest.approx(.72) and y['risk'][2]==0


def test_no_amplification_without_new_seeds():
    rng=np.random.default_rng(4);a=edges(60,3,8)
    for t in range(1,60):
        k=min(t,8); x=rng.random((3,k));x/=x.sum(-1,keepdims=True);a[t,0,:,:k]=x
    s=np.zeros(60);s[0]=.8;cfg=Config(window=8)
    r=propagate(s,a,cfg,detail=True)['channel_risk']
    assert r.max()<=.800001
    for t in range(1,60):assert (r[t]<=.9*r[max(0,t-8):t].max()+1e-6).all()


def test_head_identity_preserved_but_relay_may_switch_heads():
    a=edges(3);a[1,0,0,0]=1;a[2,0,1,0]=1
    r=propagate([1,0,0],a,Config(window=4,channel_quantile=1),detail=True)
    assert r['risk'][1]==pytest.approx(.9) and r['risk'][2]==pytest.approx(.81)
    assert r['channel_risk'][1].tolist()==pytest.approx([.9,0])


def test_mass_matched_controls_preserve_mass_and_valid_endpoints():
    rng=np.random.default_rng(6);a=edges(20,3,16)
    for t in range(1,20):a[t,:,:,:min(t,16)]=rng.random((1,3,min(t,16)))/32
    flat=validate_edges(a,20,16)
    for mode in ('uniform','permuted'):
        ctrl=controlled_edges(flat,mode,9)
        np.testing.assert_allclose(ctrl.sum(-1),flat.sum(-1))
        for t in range(16): assert not ctrl[t,:,t:].any()
    perm=controlled_edges(flat,'permuted',9)
    np.testing.assert_array_equal(perm[:,:,0],flat[:,:,0])
    assert not np.array_equal(perm[10],flat[10])


@pytest.mark.parametrize('bad', ['future','negative','oversum','nan'])
def test_invalid_edges_rejected(bad):
    a=edges(3)
    if bad=='future':a[0,0,0,0]=.1
    if bad=='negative':a[1,0,0,0]=-.1
    if bad=='oversum':a[2,0,0,:2]=.7
    if bad=='nan':a[1,0,0,0]=np.nan
    with pytest.raises(ValueError):propagate([0,0,0],a,Config(window=4))


def test_only_roundoff_overshoot_corrected():
    a=edges(3);a[1,:,:,0]=1.0000001;a[2,:,:,0]=.25
    b=validate_edges(a,3,4)
    assert np.all(b[1].sum(-1)==1) and np.all(b[2].sum(-1)==.25)


class Attention(nn.Module):
    """Llama-shaped test double using actual Torch attention ops, NOT HF/model weights."""
    def __init__(self,d=16,h=4,kv=2):
        super().__init__();self.head_dim=d//h;self.scaling=self.head_dim**-.5;self.h=h;self.kv=kv
        for name,n in [('q',h),('k',kv),('v',kv)]:setattr(self,name+'_proj',nn.Linear(d,n*self.head_dim,bias=False))
        self.o_proj=nn.Linear(d,d,bias=False)
    def forward(self,hidden_states,position_embeddings,attention_mask=None):
        b,s,_=hidden_states.shape
        q=self.q_proj(hidden_states).reshape(b,s,self.h,-1).transpose(1,2)
        k=self.k_proj(hidden_states).reshape(b,s,self.kv,-1).transpose(1,2)
        v=self.v_proj(hidden_states).reshape(b,s,self.kv,-1).transpose(1,2)
        c,si=(x.unsqueeze(1) for x in position_embeddings)
        def rot(x):a,b=x.chunk(2,-1);return torch.cat([-b,a],-1)
        q=q*c+rot(q)*si;k=k*c+rot(k)*si
        k=k.repeat_interleave(self.h//self.kv,1);v=v.repeat_interleave(self.h//self.kv,1)
        context=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o_proj(context.transpose(1,2).reshape(b,s,-1)),None


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.norm=nn.LayerNorm(16);self.mlp=nn.Linear(16,16)
    def forward(self,x,pos):
        x=x+self.self_attn(hidden_states=self.norm(x),position_embeddings=pos)[0]
        return x+F.silu(self.mlp(x))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(64,16);self.layers=nn.ModuleList([Block(),Block()]);self.norm=nn.LayerNorm(16)
    def forward(self,input_ids,**kwargs):
        x=self.embedding(input_ids);n=x.shape[1]
        angle=torch.arange(n,device=x.device)[:,None]*torch.tensor([.1,.2,.1,.2],device=x.device)[None]
        pos=(angle.cos()[None].to(x.dtype),angle.sin()[None].to(x.dtype))
        for layer in self.layers:x=layer(x,pos)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=Backbone();self.lm_head=nn.Linear(16,64,bias=False)
        self.config=SimpleNamespace(model_type='llama',num_attention_heads=4,num_key_value_heads=2,
                                    hidden_size=16,max_position_embeddings=128)
    def get_input_embeddings(self):return self.model.embedding


@pytest.mark.parametrize('chunk',[1,3,40])
def test_qk_capture_reproduces_torch_attention_and_hook_does_not_modify(chunk):
    torch.manual_seed(23);model=TinyModel().eval();r=row(n=16)
    x=torch.tensor([r['token_ids'][:-1]])
    before=model.model(x).last_hidden_state.detach()
    a=capture_response(model,r,window=4,chunk=chunk,logit_chunk=3)
    after=model.model(x).last_hidden_state.detach()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    assert a['local_attention'].shape==(16,2,4,4)
    assert not a['local_attention'][0].any()
    assert a['attention_replay_relative_error'].max()<1e-6
    expected=model.lm_head(before[0,3:]).float().log_softmax(-1)
    np.testing.assert_allclose(a['entropy'],(-(expected.exp()*expected).sum(-1)).detach().numpy(),rtol=1e-6)
    assert all(len(m._forward_hooks)==0 for m in model.modules())


def test_chunk_prefix_and_lag1_alignment():
    torch.manual_seed(24);model=TinyModel().eval();r=row(n=16)
    a=capture_response(model,r,window=4,chunk=1)
    b=capture_response(model,r,window=4,chunk=16)
    np.testing.assert_allclose(a['local_attention'],b['local_attention'],atol=1e-7)
    rr=copy.deepcopy(r);rr['token_ids']=rr['token_ids'][:4+5];rr['offsets']=rr['offsets'][:5]
    short=capture_response(model,rr,window=4,chunk=2)
    np.testing.assert_allclose(a['local_attention'][:5],short['local_attention'],atol=1e-7)
    assert a['local_attention'][1,:,:,0].min()>0
    assert not a['local_attention'][1,:,:,1:].any()


def test_hf_tiny_llama_optional():
    transformers=pytest.importorskip('transformers',reason='local environment has no transformers; Torch hook tests above still run')
    from transformers import LlamaConfig,LlamaForCausalLM
    cfg=LlamaConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,
                    num_key_value_heads=2,vocab_size=64,max_position_embeddings=128)
    cfg._attn_implementation='sdpa';model=LlamaForCausalLM(cfg).eval()
    a=capture_response(model,row(n=16),window=4,chunk=3)
    assert a['attention_replay_relative_error'].max()<1e-4


def test_capture_removes_hooks_after_failure():
    model=TinyModel().eval();r=row(n=16);r['source_mask']=[True]
    with pytest.raises(ValueError):capture_response(model,r,window=4)
    assert all(len(m._forward_hooks)==0 and len(m._forward_pre_hooks)==0 for m in model.modules())


@pytest.mark.parametrize('weighted',[False,True])
def test_fast_metrics_agree_with_sklearn(weighted):
    rng=np.random.default_rng(18)
    for _ in range(12):
        y=rng.integers(0,2,100);s=rng.integers(0,5,100);w=rng.random(100) if weighted else None
        r=Ranking(y,s).measure(w)
        assert r['auroc']==pytest.approx(roc_auc_score(y,s,sample_weight=w))
        assert r['ap']==pytest.approx(average_precision_score(y,s,sample_weight=w))


def test_zero_weight_ties_and_empty_metrics():
    m=Ranking([0,1,0,1],[0,1,2,3]);r=m.measure([0,0,1,1])
    assert r['auroc']==1 and r['ap']==1
    assert Ranking([],[]).measure()['auroc'] is None
    assert Ranking([1,1],[0,1]).measure()['auroc'] is None


def pipeline_fixture(tmp):
    root=tmp/'output';cache=root/'capture';cache.mkdir(parents=True)
    rows=[row(str(i),'train' if i<8 else 'test',n=20) for i in range(10)]
    rng=np.random.default_rng(5)
    for i,r in enumerate(rows):
        n=len(r['offsets']);a=edges(n,2,4)
        for t in range(1,n):a[t,:,:,:min(t,4)]=rng.random((1,2,min(t,4)))*.15
        save_npz(cache/(r['id']+'.npz'),entropy=rng.uniform(0,6,n),negative_margin=rng.uniform(-4,0,n),
                 local_attention=a,source_mass=np.ones((n,1,2))*.25,
                 history_mass=a.sum(-1),remote_mass=np.zeros((n,1,2)),
                 offsets=r['offsets'],input_identity=np.asarray(digest(r)))
    write_json(cache/'records.json',rows);write_json(cache/'complete.json',{'complete':True})
    annotations=tmp/'annotations.jsonl'
    gold=[dict(id=r['id'],source_id=r['source_id'],split=r['official_split'],response='abcdefghijklmnopqrst',
               labels=[dict(start=2,end=5),dict(start=10,end=13)] if r['id']=='8' else []) for r in rows]
    annotations.write_text('\n'.join(json.dumps(g) for g in gold))
    return root,rows,annotations


def test_end_to_end_label_free_scoring_then_separate_evaluation(tmp_path):
    root,rows,annotations=pipeline_fixture(tmp_path)
    score(rows,root,Config(window=4,min_reference=2),alarm_budget=.1)
    freeze=json.loads((root/'scores/prediction_freeze.json').read_text())
    assert freeze['labels_read'] is False
    original={r['id']:(root/'scores'/f"{r['id']}.npz").read_bytes() for r in rows}
    report=evaluate(rows,root,annotations,bootstrap=3)
    v=report['groups']['ALL']['views']
    assert v['all_error']['positives']==6
    assert v['first_error_until_first']['positives']==1
    assert v['span_onset_full_stream']['positives']==2
    assert v['continuation_vs_normal']['positives']==4
    # Evaluation weights are fixed by the full answer, never gold-truncated length.
    assert v['first_error_until_first']['source_fixed_full_answer']['reuse']['prevalence']==pytest.approx(1/23)
    for r in rows:assert original[r['id']]==(root/'scores'/f"{r['id']}.npz").read_bytes()
    report2=evaluate(rows,root,annotations,bootstrap=3)
    assert report==report2


def test_resume_no_refit_and_reject_changed_config(tmp_path):
    root,rows,_=pipeline_fixture(tmp_path);cfg=Config(window=4,min_reference=2)
    score(rows,root,cfg)
    old=(root/'scores/reference.json').read_bytes();score(rows,root,cfg)
    assert (root/'scores/reference.json').read_bytes()==old
    with pytest.raises(ValueError):score(rows,root,Config(window=4,survival=.8))


def test_no_labels_needed_for_complete_inference(tmp_path):
    root,rows,annotations=pipeline_fixture(tmp_path);annotations.unlink()
    score(rows,root,Config(window=4))
    assert (root/'scores/prediction_freeze.json').exists()
    with pytest.raises(FileNotFoundError):evaluate(rows,root,annotations)


def test_manifest_filter_and_no_annotation_fields(tmp_path):
    rows=[{**row('a'), 'labels':'must_not_transfer'}]
    (tmp_path/'inputs.jsonl').write_text(json.dumps(rows[0]))
    out=load_roster(tmp_path,generators='all')
    assert 'labels' not in out[0] and out[0]['id']=='a'


def test_score_is_fully_unaffected_by_gold_fields():
    r=row(n=20);cfg=Config(window=4,min_reference=1)
    ref=fit_reference([r],lambda _:np.arange(20.),cfg)
    trace=dict(entropy=np.arange(20.),negative_margin=np.zeros(20),local_attention=edges(20),
               source_mass=np.zeros((20,1,2)),history_mass=np.zeros((20,1,2)),remote_mass=np.zeros((20,1,2)))
    a,_=score_trace(trace,r,ref);r['labels']=[{'start':0,'end':20}]
    b,_=score_trace(trace,r,ref)
    for k in a:np.testing.assert_array_equal(a[k],b[k])


def test_default_cli_not_supervised():
    a=parser().parse_args(['--output','x'])
    assert a.phase=='all' and a.annotations is None and a.generators=='llama-2-7b-chat'
    assert not hasattr(a,'fit_labels') and not hasattr(a,'s10_predictions')


def test_bfloat16_readonly_capture():
    torch.manual_seed(91);model=TinyModel().to(torch.bfloat16).eval()
    data=capture_response(model,row(n=12),window=4,chunk=3)
    assert data['attention_replay_relative_error'].max()<.02
    assert data['local_attention'].dtype==np.float32


def test_coarse_quantile_is_not_attention_averaging():
    a=edges(2,4,4);a[1,0,:,0]=[0.,0.,0.,1.]
    r=propagate([1.,0.],a,Config(window=4,channel_quantile=1),detail=True)
    assert r['risk'][1]==pytest.approx(.9)
    assert r['channel_inherited'][1].tolist()==pytest.approx([0,0,0,.9])


def test_ablation_onehop_root_is_actual_seed_position():
    a=edges(4);a[1:,:,:,0]=1
    r=propagate([1.,0.,0.,0.],a,Config(window=4),recursive=False,detail=True)
    assert r['dominant_origin'][1].tolist()==[0,0]
    assert (r['dominant_origin'][2:]==-1).all()


def test_actual_score_cli_phase_never_opens_annotations(tmp_path):
    root,rows,annotations=pipeline_fixture(tmp_path)
    population=tmp_path/'population';population.mkdir()
    (population/'inputs.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    a=parser().parse_args(['--population',str(population),'--output',str(root),'--phase','score',
                          '--generators','all','--window','4','--resume','--annotations','/no/labels.jsonl'])
    run(a)
    assert (root/'scores/prediction_freeze.json').exists()
    assert not (root/'evaluation.json').exists()


def test_global_answer_max_cannot_gain_from_contractive_propagation():
    rng=np.random.default_rng(47)
    a=edges(80,4,8)
    for t in range(1,80):
        k=min(t,8);v=rng.random((4,k));v/=v.sum(-1,keepdims=True)*1.1
        a[t,0,:,:k]=v
    seeds=rng.random(80);seeds[seeds<.8]=0
    r=propagate(seeds,a,Config(window=8))['risk']
    assert r.max()==pytest.approx(seeds.max())
    for threshold in [0,.1,.5,.9,1]:
        assert bool((r>threshold).any())==bool((seeds>threshold).any())


def test_ambiguous_transport_command_refuses_supervised_training():
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    completed=subprocess.run([sys.executable,str(root/'main.py'),'transport'],capture_output=True,text=True)
    assert completed.returncode!=0
    assert 'supervised' in completed.stderr
