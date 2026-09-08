import numpy as np
import torch
import pytest

from experiments.reanchor_flow.message_dag.events import EventConfig, row_change, detect_layer, scan
from experiments.reanchor_flow.message_dag.event_trace import trace_events, route_hops
from experiments.reanchor_flow.message_dag.cache import NativeCache
from experiments.reanchor_flow.message_dag.differential import DifferentialLayer, rms_jvp
from experiments.reanchor_flow.tests.test_message_lineage import capture_fixture


def test_window_aging_does_not_create_lookback_and_old_response_is_eligible():
    n=14;special=np.zeros(n,bool);special[0]=True
    a=np.zeros((1,n),np.float32);a[0,5]=1
    change=row_change(a,a,8,special,2)
    assert change['remote_gain'][0]==0  # source5 became remote just by aging
    previous=np.zeros_like(a);previous[0,9]=1
    current=np.zeros_like(a);current[0,5]=.8;current[0,10]=.2
    rows=np.array([9,10]);array=np.stack((previous,current),axis=1)
    found=detect_layer([(0,2,array)],rows,special,4,EventConfig(window=2))
    assert found['event'][0,1]
    assert found['peak_source'][0,1]==5  # old RESPONSE, not prompt
    assert found['time_tv'][0,1]==pytest.approx(1.)
    # Merely increasing a special sink cannot become a global factual root.
    current[:]=0;current[0,0]=.8;current[0,10]=.2
    found=detect_layer([(0,2,np.stack((previous,current),axis=1))],rows,special,4,EventConfig(window=2))
    assert not found['event'].any()


def test_streaming_chunks_preserve_all_physical_heads():
    rng=np.random.default_rng(4);rows=np.arange(4,12);special=np.zeros(12,bool)
    a=rng.random((3,len(rows),12)).astype(np.float32)
    a*=np.arange(12)[None,None]<=rows[None,:,None];a/=a.sum(-1,keepdims=True)
    x=detect_layer([(0,8,a)],rows,special,5,EventConfig(2,.05,.2))
    y=detect_layer([(0,3,a[:,:3]),(3,8,a[:,3:])],rows,special,5,EventConfig(2,.05,.2))
    for k in x: np.testing.assert_allclose(x[k],y[k],equal_nan=True)
    assert x['event'].shape==(3,8)


def test_hop_partition_is_additive_and_requires_two_crossings_for_multihop():
    state=torch.zeros(3,1,4,2);same=torch.zeros_like(state);cross=torch.zeros_like(state)
    cross[0,0,2,0]=3
    post=route_hops(state,same,cross)
    assert post[1,0,2,0]==3 and not post[2].any()
    cross.zero_();cross[1,0,3,0]=2
    final=route_hops(post,same,cross)
    assert final[2,0,3,0]==2
    torch.testing.assert_close(final.sum(0),(post+same+cross).sum(0))


def test_rms_derivative_includes_denominator():
    x=torch.tensor([[1.,2.,3.]])
    w=torch.tensor([.8,1.,1.2]);dx=x.clone();eps=1e-6
    actual=rms_jvp(dx,x,w,eps)
    assert actual.norm()<1e-5  # radial perturbations mostly disappear in RMSNorm
    assert (dx*w/torch.sqrt(x.square().mean()+eps)).norm()>1


@pytest.mark.parametrize('qk_scale',[1,8])
def test_event_dag_matches_internal_message_finite_difference(tmp_path,qk_scale):
    from transformers import LlamaForCausalLM
    torch.set_num_threads(2)
    path,trace,weights=capture_fixture(tmp_path,qk_scale=qk_scale)
    coord=np.array([[0,1,2]])
    with NativeCache(path,weights) as cache:
        event=trace_events(cache,coord,window=2,query_chunk=3)[0]
        op=DifferentialLayer(cache,0,chunk=3)
        seed,_=op.remote_seed(1,2,2)
    model=LlamaForCausalLM.from_pretrained(weights.directory).eval()
    model.config._attn_implementation='eager'
    ids=torch.tensor(trace['token_ids'])[None]
    position=int(trace['row_position'][2]);epsilon=.03
    q=torch.tensor(trace['row_position'][:-1])
    pos=torch.tensor(event['positive_id']);neg=torch.tensor(event['negative_id'])
    def forward(strength):
        def perturb(module,args,output):
            value=output[0].clone();value[:,position]+=strength*seed
            return (value,*output[1:])
        handle=model.model.layers[0].self_attn.register_forward_hook(perturb)
        try:
            logits=model(ids,use_cache=False).logits[0]
            return logits[q,pos]-logits[q,neg]
        finally: handle.remove()
    with torch.no_grad(): finite=((forward(epsilon)-forward(-epsilon))/(2*epsilon)).numpy()
    predicted=event['margin_response'][0].sum(0)
    np.testing.assert_allclose(predicted,finite,atol=2e-6,rtol=.015)
    # Independent autograd oracle is test-only. The production audit uses
    # analytic Jacobians and never builds a training/backpropagation graph.
    exact=torch.autograd.functional.jacobian(forward,torch.tensor(0.)).detach().numpy()
    np.testing.assert_allclose(predicted,exact,atol=2e-8,rtol=1e-4)
    with torch.no_grad(): baseline=forward(0.).numpy()
    np.testing.assert_allclose(event['baseline_margin'],baseline,atol=1e-6,rtol=1e-5)
    assert not event['margin_response'][...,trace['row_position'][:-1]<position].any()
    assert np.max(np.abs(event['margin_response'][0,2]))>0
    assert not np.allclose(event['margin_response'][0],event['margin_response'][1],atol=1e-10)
    assert not np.allclose(event['margin_response'][0],event['margin_response'][2],atol=1e-10)
    assert np.abs(event['margin_response'][0,2,event['target_position']<position+3]).max()<2e-9
    with NativeCache(path,weights) as cache:
        i=4
        reverse=[dict(target=int(event['target_position'][i]),positive_id=int(event['negative_id'][i]),
                      negative_id=int(event['positive_id'][i]))]
        changed=trace_events(cache,coord,window=2,contrasts=reverse)[0]
    np.testing.assert_allclose(changed['margin_response'][...,i],-event['margin_response'][...,i],atol=2e-9,rtol=1e-4)
    assert changed['baseline_margin'][i]==pytest.approx(-event['baseline_margin'][i])
    assert changed['explicit_contrast'].sum()==1


def test_no_events_is_valid_and_scan_does_not_read_label_files(tmp_path):
    path,trace,weights=capture_fixture(tmp_path)
    np.savez_compressed(path.with_suffix('.labels.npz'),labels=np.ones(8,int))
    x=scan(path,EventConfig(2,.01,.1),chunk=3)
    np.savez_compressed(path.with_suffix('.labels.npz'),labels=np.zeros(8,int))
    y=scan(path,EventConfig(2,.01,.1),chunk=8)
    for k in ('event_index','event','remote_gain','time_tv'):
        np.testing.assert_allclose(x[k],y[k],equal_nan=True,atol=2e-7)
    with NativeCache(path,weights) as cache: assert trace_events(cache,[])==[]


def test_same_position_sites_combine_at_native_layers_and_batches_do_not_mix(tmp_path):
    path,trace,weights=capture_fixture(tmp_path)
    sites=np.array([[0,1,2],[1,2,2],[0,3,4]])
    with NativeCache(path,weights) as cache:
        together=trace_events(cache,sites,window=2,query_chunk=3)
        individuals=[trace_events(cache,[s],window=2,query_chunk=8)[0] for s in sites]
    assert len(together)==2 and len(together[0]['event_sites'])==2
    np.testing.assert_allclose(together[0]['margin_response'],
                               individuals[0]['margin_response']+individuals[1]['margin_response'],atol=2e-9,rtol=1e-4)
    np.testing.assert_allclose(together[1]['margin_response'],individuals[2]['margin_response'],atol=2e-9,rtol=1e-4)


def test_same_reference_mlp_output_can_have_opposite_message_derivatives():
    # A smooth native SwiGLU counterexample, not empirical hallucination labels.
    # At x=e2, up(x)=0 so both MLPs output zero and have identical forward A.
    # Yet the message e1 is transmitted into (or against) the e2 reader direction.
    import torch.nn.functional as F
    x=torch.tensor([[0.,1.]])
    delta=torch.tensor([[[1.,0.]]])
    eps=1e-6
    op=object.__new__(DifferentialLayer)
    op.post=x;op.eps=eps
    op.w={'post_norm':torch.ones(2),'up':torch.tensor([[1.,0.]]),
          'gate':torch.tensor([[0.,1.]]),'down':torch.tensor([[0.],[1.]])}
    z=x/torch.sqrt(x.square().mean(-1,keepdim=True)+eps)
    op.up=F.linear(z,op.w['up']);gate=F.linear(z,op.w['gate'])
    op.activation=F.silu(gate);sigmoid=gate.sigmoid()
    op.silu_prime=sigmoid*(1+gate*(1-sigmoid))
    positive=op.mlp_jvp(delta)
    baseline=F.linear(op.activation*op.up,op.w['down'])
    op.w['down']=-op.w['down']
    negative=op.mlp_jvp(delta)
    torch.testing.assert_close(baseline,F.linear(op.activation*op.up,op.w['down']))
    assert baseline.norm()==0 and positive[0,0,1]>0 and negative[0,0,1]<0


def test_full_scope_pipeline_resume_missing_coverage_and_offline_report(tmp_path):
    import json
    import shutil
    from experiments.reanchor_flow.message_dag.event_run import parser,run
    fixture=tmp_path/'fixture';fixture.mkdir()
    path,trace,weights=capture_fixture(fixture,ids=[1,3,5,7,9,11,13,15,17,19,21,23,25,2,4,6,8,10,12,14],qk_scale=8)
    audit=tmp_path/'audit';audit.mkdir();entries=[]
    for split in ('train','test'):
        for task in ('QA','Summary','Data2txt'):
            dest=audit/split/task/'1.npz';dest.parent.mkdir(parents=True)
            for suffix in ('.npz','.qk.npz','.history.npz','.states.npz'):
                shutil.copyfile(path.with_suffix(suffix),dest.with_suffix(suffix))
            labels=np.zeros(15,int);labels[10]=1  # synthetic labels only test joins
            np.savez_compressed(dest.with_suffix('.labels.npz'),labels=labels)
            entries.append(dict(split=split,task_type=task,sample_id='1',source_id='one',
                                path=str(dest.relative_to(audit)),response_tokens=15))
    missing={**entries[0],'path':'train/QA/missing.npz','sample_id':'missing'}
    manifest=dict(audit_schema=3,settings=dict(model=str(weights.directory),save_states=True),samples=entries+[missing])
    (audit/'index.json').write_text(json.dumps(manifest))
    out=tmp_path/'events'
    command=['--audit',str(audit),'--output',str(out),'--device','cpu','--window','2',
             '--gain','.002','--local-floor','.05','--event-batch','3','--bootstrap','10']
    with pytest.raises(ValueError,match='requested captures complete'): run(parser().parse_args(command))
    result=run(parser().parse_args(command+['--completed-only']))
    assert result['scanned']==6 and result['native_coverage']['skipped_samples']==1
    assert result['traced']==result['events']>0
    assert set(r['sample'].split('/')[1] for r in result['samples'])=={'QA','Summary','Data2txt'}
    files=list(out.rglob('event_*.npz'));before={p:p.stat().st_mtime_ns for p in files}
    # Changing only labels cannot alter the saved graph or its selection.
    for dest in audit.rglob('*.labels.npz'): np.savez_compressed(dest,labels=np.zeros(15,int))
    rerun=run(parser().parse_args(command+['--completed-only']))
    assert before=={p:p.stat().st_mtime_ns for p in files}
    assert all(r['H']==0 for r in rerun['samples'])
    shutil.rmtree(audit);shutil.rmtree(fixture)
    offline=run(parser().parse_args(['--phase','evaluate','--output',str(out),'--bootstrap','0']))
    assert offline['traced']==result['traced']
    assert (out/'gallery.html').exists() and list(out.rglob('preview.html'))
