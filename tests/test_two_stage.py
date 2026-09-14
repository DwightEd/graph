import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from structural_detector.transport import propagate,mixture,onset_features,continuation_features
from structural_detector.local_capture import attention_band,replay,write
from structural_detector.two_stage import (choose_channels,normal_answer_threshold,read_labels,
                                           train_and_evaluate,train_entry)


def chain(t=20,k=2,mass=.8,window=4):
    w=np.zeros((t,k,window));w[1:,:,0]=mass
    return w


def test_zero_seed_never_creates_error():
    u,r,p=propagate(np.zeros(20),chain())
    assert not u.any() and not r.any() and np.all(p==-1)


def test_one_seed_has_multihop_and_decays_without_local_renormalization():
    a=np.zeros(20);a[0]=1
    u,r,_=propagate(a,chain())
    np.testing.assert_allclose(r[:,0],.8**np.arange(20))
    assert r[15,0]>0   # longer than the stored lag window
    assert u[0,0]==0
    direct=propagate(a,chain(),'one_hop')[0]
    assert direct[1,0]==.8 and direct[2,0]==0


def test_no_edges_is_no_inheritance():
    a=np.linspace(0,1,20);u,r,_=propagate(a,chain(),'no_edges')
    assert not u.any();np.testing.assert_allclose(r[:,0],a)


def test_source_switch_breaks_chain():
    w=chain();w[5:]=0;a=np.zeros(20);a[0]=1
    r=propagate(a,w)[1];assert r[4,0]>0 and not r[5:].any()


def test_endpoints_not_just_local_mass_matter():
    a=np.array([1,0,0,0.]);w=np.zeros((4,1,3));w[3,0,2]=.9
    real=propagate(a,w)[0];uniform=propagate(a,w,'uniform')[0]
    assert real[3,0]==.9 and uniform[3,0]==pytest.approx(.3)


def test_different_heads_not_averaged_before_propagation():
    a=np.array([1,0,0.]);w=np.zeros((3,2,2));w[1,0,0]=1;w[2,0,0]=1
    u,r,_=propagate(a,w)
    assert u[2,0]==1 and u[2,1]==0 and r.shape==(3,2)


@pytest.mark.parametrize('mode',['real','uniform','permuted','one_hop','no_edges'])
def test_causal_prefix_extension(mode):
    rng=np.random.default_rng(7);w=rng.random((20,3,4));w/=w.sum(-1,keepdims=True)*1.4
    for t in range(4):w[t,:,t:]=0
    a=rng.random(20);full=propagate(a,w,mode)
    for n in (1,3,8,19):
        short=propagate(a[:n],w[:n],mode)
        for aa,bb in zip(full,short):np.testing.assert_allclose(aa[:n],bb)


@pytest.mark.parametrize('bad',['mass','past','seed','nan'])
def test_invalid_graph_rejected(bad):
    a=np.zeros(5);w=chain(5)
    if bad=='mass':w[1,:,0]=1.1
    if bad=='past':w[0,:,0]=.1
    if bad=='seed':a[0]=1.1
    if bad=='nan':w[2,0,0]=np.nan
    with pytest.raises(ValueError):propagate(a,w)


def test_mixture_is_probability_not_unbounded_or_cumsum():
    a=np.array([0,.2,1]);c=np.array([.9,.5,0])
    np.testing.assert_allclose(mixture(a,c),[.9,.6,1])
    with pytest.raises(ValueError):mixture(a,np.ones(3)*2)


def test_entropy_seed_does_not_use_future_or_margin():
    v=np.ones((4,5));v[:,0]=[1,4,2,5]
    np.testing.assert_allclose(onset_features(v),[[1,0],[4,3],[2,-2],[5,3]])
    for end in range(1,4):np.testing.assert_array_equal(onset_features(v[:end]),onset_features(v)[:end])
    v[:,1:]=999;assert onset_features(v)[1,1]==3


def test_hard_mass_control_keeps_dimension():
    a=np.array([1,0,0.]);w=chain(3);source=np.ones((3,2))*.2
    x,u,_,_=continuation_features(a,w,source,'real')
    null=continuation_features(a,w,source,'no_edges')[0]
    assert x.shape==(3,6)
    np.testing.assert_array_equal(x[:,2:],null[:,2:]);assert not null[:,:2].any()


def test_attention_alignment_including_preceding_token_self():
    torch=pytest.importorskip('torch');p,t,n=3,4,6
    a=torch.zeros((1,2,n,n))
    for q in range(n):a[:,:,q,:q+1]=1/(q+1)
    band,e,remote=attention_band(a,p,t,3,[False,True,True])
    assert not band[0].any()
    assert band[1,0,0]==pytest.approx(1/4)  # row P forecasts response1, self is response0
    assert band[3,0,2]==pytest.approx(1/6)
    np.testing.assert_allclose(e[:,0],[2/3,2/4,2/5,2/6])
    assert not remote.any()


def test_remote_boundary_agrees_with_population_code():
    torch=pytest.importorskip('torch');p,t=3,20;n=p+t-1
    a=torch.zeros((1,1,n,n))
    for q in range(n):a[:,:,q,:q+1]=1/(q+1)
    band,e,remote=attention_band(a,p,t,16,[0,1,1])
    assert remote[16,0]==0 and remote[17,0]==pytest.approx(1/20)
    assert band[17,0].sum()+remote[17,0]+3/20==pytest.approx(1)


def toy_model():
    torch=pytest.importorskip('torch');nn=torch.nn
    class Attention(nn.Module):
        def __init__(self):
            super().__init__();self.q=nn.Linear(8,8);self.k=nn.Linear(8,8);self.v=nn.Linear(8,8);self.o=nn.Linear(8,8)
        def forward(self,x):
            b,n,d=x.shape
            q=self.q(x).view(b,n,2,4).transpose(1,2);k=self.k(x).view(b,n,2,4).transpose(1,2);v=self.v(x).view(b,n,2,4).transpose(1,2)
            s=q@k.transpose(-1,-2)/2;s=s.masked_fill(torch.ones((n,n),dtype=torch.bool).triu(1),-torch.inf)
            a=s.softmax(-1);return self.o((a@v).transpose(1,2).reshape(b,n,8)),a
    class Layer(nn.Module):
        def __init__(self):super().__init__();self.self_attn=Attention();self.ff=nn.Linear(8,8)
        def forward(self,x):
            y,a=self.self_attn(x);return x+y+torch.tanh(self.ff(x+y)),a
    class Decoder(nn.Module):
        def __init__(self):super().__init__();self.emb=nn.Embedding(32,8);self.layers=nn.ModuleList([Layer(),Layer()]);self.norm=nn.LayerNorm(8)
        def forward(self,input_ids,**kw):
            x=self.emb(input_ids);att=[]
            for layer in self.layers:x,a=layer(x);att.append(a)
            return SimpleNamespace(last_hidden_state=self.norm(x),attentions=att)
    class LM(nn.Module):
        def __init__(self):super().__init__();self.model=Decoder();self.lm_head=nn.Linear(8,32,bias=False)
    torch.manual_seed(3);return LM().eval()


def test_replay_hook_does_not_change_logits_and_removes_handles():
    torch=pytest.importorskip('torch');model=toy_model()
    record=dict(prompt_length=3,token_ids=[1,2,3,4,5,6,7],offsets=[[0,1],[1,2],[2,3],[3,4]],source_mask=[False,True,True])
    with torch.no_grad():
        baseline=model.model(torch.tensor([record['token_ids'][:-1]]));z=model.lm_head(baseline.last_hidden_state[0,2:]);lp=z.log_softmax(-1)
    result=replay(model,record,window=3,logit_chunk=2)
    np.testing.assert_allclose(result['values'][:,0],(-(lp.exp()*lp).sum(-1)).detach().numpy(),rtol=1e-6)
    top=z.topk(2,dim=-1).values
    np.testing.assert_allclose(result['values'][:,1],(top[:,1]-top[:,0]).detach().numpy(),atol=1e-6)
    assert result['edges'].shape==(4,4,3)
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
    with torch.no_grad():after=model.model(torch.tensor([record['token_ids'][:-1]])).last_hidden_state
    torch.testing.assert_close(after,baseline.last_hidden_state)


def fixture(tmp,answers=32):
    root=tmp/'features';root.mkdir();rows=[];gold=[]
    rng=np.random.default_rng(2)
    for i in range(answers):
        rid=str(i);text='abcdefgh';sp='train' if i<24 else 'test'
        offsets=[[j,j+1] for j in range(8)];y=(i%2==0)
        v=rng.random((8,5));v[:,0]=.5
        if y:v[2,0]=4
        edges=chain(8,2,window=3).astype(np.float32)
        r=dict(id=rid,source_id=rid,task='QA',generator='fixture',official_split=sp,
               response_sha256=hashlib.sha256(text.encode()).hexdigest())
        rows.append(r);gold.append(dict(id=rid,source_id=rid,split=sp,response=text,labels=[dict(start=2,end=6)] if y else []))
        np.savez_compressed(root/(rid+'.npz'),values=v,edges=edges,evidence=np.ones((8,2))*.1,offsets=offsets,channels=[[0,0],[0,1]])
    write(root/'records.json',rows);write(root/'complete.json',dict(complete=True))
    ann=tmp/'response.jsonl';ann.write_text('\n'.join(json.dumps(g) for g in gold))
    return root,ann


def test_threshold_controls_answers_not_negative_tokens():
    rows=[dict(id=str(i)) for i in range(39)];labels={str(i):(np.zeros(100,bool),np.zeros(100,bool)) for i in range(39)}
    p={str(i):np.r_[np.zeros(99),i/40] for i in range(39)}
    t=normal_answer_threshold(p,labels,rows)
    assert t['normal_calibration_answers']==39 and t['threshold']>0.8
    assert normal_answer_threshold(p,labels,rows[:1])['threshold'] is None


def test_training_loader_never_parses_test_json(tmp_path):
    p=tmp_path/'labels';text='abc';h=hashlib.sha256(text.encode()).hexdigest()
    p.write_text('{"id":"1","source_id":"1","split":"train","response":"abc","labels":[]}\n{"id":"2", MALFORMED_UNREAD_TEST}\n')
    r=[dict(id='1',source_id='1',official_split='train',response_sha256=h,offsets=[[0,3]])]
    assert not read_labels(p,r,{'1'})['1'][0].any()


def test_label_free_channel_choice_not_test_driven():
    rows=[dict(id='1',source_id='1'),dict(id='2',source_id='2')]
    d={'1':dict(edges=chain(8,2)),'2':dict(edges=chain(8,2))}
    d['1']['edges'][:,1]=0;d['2']['edges'][:,1]=0
    assert choose_channels(rows,d,1).tolist()==[0]


def test_full_two_stage_pipeline_freezes_before_oracles(tmp_path):
    root,ann=fixture(tmp_path);out=tmp_path/'out'
    train_and_evaluate(root,ann,out,max_channels=2,bootstrap=0,oracle=True)
    group=out/'QA__fixture'
    assert json.loads((group/'prediction_freeze.json').read_text())['test_labels_read'] is False
    result=json.loads((group/'evaluation.json').read_text())
    assert result['views']['continuation_vs_normal']['metrics']['error__real']['eligible_positives']==12
    assert result['views']['first_error_until_first']['metrics']['onset']['eligible_positives']==4
    assert 'USES TEST LABELS' in json.loads((group/'oracle_diagnostic.json').read_text())['warning']
    with np.load(group/'24.npz') as f:
        assert all('oracle' not in k for k in f.files)
        assert np.all(f['error__real']>=f['onset'])
        assert f['dominant_parent'].shape==(8,2)
    with pytest.raises(FileExistsError):train_and_evaluate(root,ann,out,max_channels=2,bootstrap=0)


def test_continuation_cannot_learn_negative_inherited_risk():
    from structural_detector.two_stage import fit_linear
    x=np.linspace(-2,2,60)[:,None];y=(x[:,0]<0).astype(int)
    fitted=fit_linear(x,y,np.ones(60),nonnegative=(0,))
    assert fitted['coef'][0]>=0


def test_test_labels_do_not_change_scores(tmp_path):
    root,ann=fixture(tmp_path)
    out1=tmp_path/'a';out2=tmp_path/'b'
    train_and_evaluate(root,ann,out1,max_channels=2,bootstrap=0,oracle=False)
    raw=list(map(json.loads,ann.read_text().splitlines()))
    for r in raw:
        if r['split']=='test':r['labels']=[dict(start=0,end=1)]
    ann.write_text('\n'.join(json.dumps(r) for r in raw))
    train_and_evaluate(root,ann,out2,max_channels=2,bootstrap=0,oracle=False)
    for rid in ('24','25','31'):
        with np.load(out1/'QA__fixture'/f'{rid}.npz') as a,np.load(out2/'QA__fixture'/f'{rid}.npz') as b:
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])


def test_source_folds_never_train_seed_on_heldout_source(tmp_path):
    from structural_detector.two_stage import splits
    root,ann=fixture(tmp_path)
    rows=json.loads((root/'records.json').read_text());sp=splits(rows);data={}
    for r in rows:
        with np.load(root/(r['id']+'.npz')) as f:
            data[r['id']]={'values':f['values']};r['offsets']=f['offsets'].tolist()
    train=[r for r in rows if sp[r['id']]=='train']
    lab=read_labels(ann,rows,{r['id'] for r in train})
    _,seeds,folds=train_entry(train,data,lab)
    held=train[0];fold=folds[held['source_id']]
    for r in train:
        if folds[r['source_id']]==fold:
            y,o=lab[r['id']];lab[r['id']]=(y,~o)
    _,changed,_=train_entry(train,data,lab)
    np.testing.assert_allclose(seeds[held['id']],changed[held['id']],atol=0,rtol=0)


def test_optional_real_transformers_tiny_llama():
    transformers=pytest.importorskip('transformers')
    torch=pytest.importorskip('torch')
    config=transformers.LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                                   num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                                   max_position_embeddings=128,attention_dropout=0.)
    config._attn_implementation='eager'
    model=transformers.LlamaForCausalLM(config).eval()
    record=dict(prompt_length=3,token_ids=[1,2,3,4,5,6],offsets=[[0,1],[1,2],[2,3]],source_mask=[False,True,True])
    with torch.inference_mode():
        z=model(torch.tensor([record['token_ids'][:-1]]),use_cache=False).logits[0,2:].float();lp=z.log_softmax(-1)
    output=replay(model,record,window=3,logit_chunk=2)
    np.testing.assert_allclose(output['values'][:,0],(-(lp.exp()*lp).sum(-1)).numpy(),rtol=1e-5)
    assert output['edges'].shape==(3,8,3)
